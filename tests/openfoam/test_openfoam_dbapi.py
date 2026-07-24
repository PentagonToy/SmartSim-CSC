from pathlib import Path
import os
import shutil
import subprocess
import tempfile

from smartsim import Experiment
from smartredis import Client


ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_CASE = ROOT_DIR / "tests" / "openfoam" / "cases" / "cavity"


with tempfile.TemporaryDirectory(prefix="openfoam-dbapi-") as tmpdir:
    case_dir = Path(tmpdir) / "cavity"
    shutil.copytree(SOURCE_CASE, case_dir)

    subprocess.run(["blockMesh", "-case", str(case_dir)], check=True)
    subprocess.run(["icoFoam", "-case", str(case_dir)], check=True)

    experiment_dir = Path(tmpdir) / "experiment"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    exp = Experiment(
        "openfoam-dbapi-test",
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
            [
                "foamSmartSimSvdDBAPI",
                "-case",
                str(case_dir),
                "-fieldName",
                "p",
                "-noZero",
            ],
            env=env,
            check=True,
        )

        client = Client(address=address, cluster=False)

        metadata_name = "foamSmartSimSvdDBAPI_metadata"
        metadata = client.get_dataset(metadata_name)
        n_times = metadata.get_meta_strings("NTimes")

        assert n_times == ["50"]

        for time_index in (1, 50):
            dataset_name = (
                f"foamSmartSimSvdDBAPI_time_index_{time_index}_mpi_rank_0"
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

        print(f"{metadata_name}: NTimes={n_times[0]}")

    finally:
        exp.stop(db)
