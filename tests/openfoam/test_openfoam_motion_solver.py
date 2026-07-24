from pathlib import Path
import os
import shutil
import subprocess
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper
from smartsim import Experiment
from smartredis import Client


ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_CASE = (
    ROOT_DIR
    / "components"
    / "openfoam-smartsim"
    / "tutorials"
    / "meshMotion"
    / "spinningDisk"
)


def create_zero_displacement_model():
    input_info = helper.make_tensor_value_info(
        "input",
        TensorProto.DOUBLE,
        [None, 2],
    )
    output_info = helper.make_tensor_value_info(
        "output",
        TensorProto.DOUBLE,
        [None, 2],
    )

    zero_displacement = helper.make_node(
        "Sub",
        inputs=["input", "input"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        [zero_displacement],
        "zero-displacement",
        [input_info],
        [output_info],
    )

    model = helper.make_model(
        graph,
        producer_name="SmartSim-CSC",
        opset_imports=[helper.make_opsetid("", 13)],
    )

    onnx.checker.check_model(model)
    return model.SerializeToString()


with tempfile.TemporaryDirectory(prefix="openfoam-motion-solver-") as tmpdir:
    case_dir = Path(tmpdir) / "spinningDisk"
    shutil.copytree(SOURCE_CASE, case_dir)
    shutil.copytree(case_dir / "0.orig", case_dir / "0")

    control_dict = case_dir / "system" / "controlDict"
    control_text = control_dict.read_text()
    control_text = control_text.replace(
        "endTime         20e-01;",
        "endTime         0.2;",
    )
    control_text = control_text.replace(
        "deltaT          1e-01;",
        "deltaT          0.1;",
    )
    control_dict.write_text(control_text)

    subprocess.run(
        ["blockMesh", "-case", str(case_dir)],
        check=True,
    )

    experiment_dir = Path(tmpdir) / "experiment"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    exp = Experiment(
        "openfoam-motion-solver-test",
        launcher="local",
        exp_path=str(experiment_dir),
    )
    db = exp.create_database(
        port=6780,
        interface="lo",
    )

    process = None

    try:
        exp.start(db, block=False)

        address = db.get_address()[0]
        client = Client(
            address=address,
            cluster=False,
        )

        client.set_model(
            "MLP",
            create_zero_displacement_model(),
            "ONNX",
            "CPU",
        )

        env = os.environ.copy()
        env["SSDB"] = address

        process = subprocess.Popen(
            ["moveDynamicMesh", "-case", str(case_dir)],
            env=env,
        )

        completed_steps = 0

        while completed_steps < 2:
            points_ready = client.poll_list_length(
                "pointsDatasetList",
                1,
                10,
                1000,
            )
            displacements_ready = client.poll_list_length(
                "displacementsDatasetList",
                1,
                10,
                1000,
            )

            if not points_ready or not displacements_ready:
                raise RuntimeError(
                    "OpenFOAM did not publish the mesh-motion datasets."
                )

            points_datasets = client.get_datasets_from_list(
                "pointsDatasetList"
            )
            displacement_datasets = client.get_datasets_from_list(
                "displacementsDatasetList"
            )

            assert len(points_datasets) == 1
            assert len(displacement_datasets) == 1

            points_dataset = points_datasets[0]
            displacement_dataset = displacement_datasets[0]

            assert points_dataset.get_tensor_names()
            assert displacement_dataset.get_tensor_names()

            client.delete_list("pointsDatasetList")
            client.delete_list("displacementsDatasetList")

            client.put_tensor(
                "model_updated",
                np.array([1.0], dtype=np.float64),
            )

            completed_steps += 1
            print(
                f"Completed mesh-motion coupling step "
                f"{completed_steps}"
            )

        return_code = process.wait(timeout=60)

        if return_code != 0:
            raise RuntimeError(
                f"moveDynamicMesh exited with code {return_code}"
            )

        end_time_index = client.get_tensor("end_time_index")
        output = client.get_tensor("outputDisplacements_0")

        assert end_time_index.shape == (1,)
        assert end_time_index[0] == 2.0

        assert output.ndim == 2
        assert output.shape[1] == 2
        assert output.dtype.name == "float64"
        assert np.allclose(output, 0.0)

        print(
            f"outputDisplacements_0: "
            f"shape={output.shape}, "
            f"maximum={np.abs(output).max()}"
        )
        print(f"end_time_index={end_time_index[0]}")

    finally:
        if process is not None and process.poll() is None:
            process.terminate()

            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        exp.stop(db)
