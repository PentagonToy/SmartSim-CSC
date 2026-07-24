from pathlib import Path
import os
import shutil
import subprocess
import tempfile

from smartsim import Experiment
from smartredis import Client


ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_CASE = ROOT_DIR / "tests" / "openfoam" / "cases" / "cavity"


with tempfile.TemporaryDirectory(prefix="openfoam-function-object-") as tmpdir:
    case_dir = Path(tmpdir) / "cavity"
    shutil.copytree(SOURCE_CASE, case_dir)

    control_dict = case_dir / "system" / "controlDict"
    with control_dict.open("a") as file:
        file.write(
            """
functions
{
    pressureToRedis
    {
        type        fieldsToSmartRedis;
        libs        ("libsmartredisFunctionObjects.so");

        clusterMode false;
        clientName  default;

        fields      (p);
        patches     (internal);
    }
}
"""
        )

    subprocess.run(["blockMesh", "-case", str(case_dir)], check=True)

    experiment_dir = Path(tmpdir) / "experiment"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    exp = Experiment(
        "openfoam-function-object-test",
        launcher="local",
        exp_path=str(experiment_dir),
    )
    db = exp.create_database(port=6780, interface="lo")

    try:
        exp.start(db, block=False)

        address = db.get_address()[0]
        env = os.environ.copy()
        env["SSDB"] = address

        subprocess.run(
            ["icoFoam", "-case", str(case_dir)],
            env=env,
            check=True,
        )

        client = Client(address=address, cluster=False)

        metadata_name = "pressureToRedis_metadata"
        metadata = client.get_dataset(metadata_name)
        end_time_index = metadata.get_meta_strings("EndTimeIndex")

        assert end_time_index == ["50"]

        for time_index in (1, 50):
            dataset_name = (
                f"pressureToRedis_time_index_{time_index}_mpi_rank_0"
            )
            dataset = client.get_dataset(dataset_name)
            tensor = dataset.get_tensor("field_name_p_patch_internal")

            assert tensor.shape == (400, 1)
            assert tensor.dtype.name == "float64"

            print(
                f"{dataset_name}: "
                f"shape={tensor.shape}, "
                f"range={tensor.min()} to {tensor.max()}"
            )

        print(
            f"{metadata_name}: "
            f"EndTimeIndex={end_time_index[0]}"
        )

    finally:
        exp.stop(db)
