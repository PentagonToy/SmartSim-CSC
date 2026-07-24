from pathlib import Path
import os
import shutil
import subprocess
import tempfile

from smartsim import Experiment
from smartredis import Client


ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_CASE = ROOT_DIR / "tests" / "openfoam" / "cases" / "cavity"


with tempfile.TemporaryDirectory(prefix="openfoam-smartsim-") as tmpdir:
    case_dir = Path(tmpdir) / "cavity"
    shutil.copytree(SOURCE_CASE, case_dir)

    subprocess.run(["blockMesh", "-case", str(case_dir)], check=True)
    subprocess.run(["icoFoam", "-case", str(case_dir)], check=True)

    exp = Experiment(
        "openfoam-smartsim-test",
        launcher="local",
        exp_path=str(Path(tmpdir) / "experiment"),
    )
    db = exp.create_database(port=6780, interface="lo")

    try:
        exp.start(db, block=False)

        address = db.get_address()[0]
        print(f"SmartSim database: {address}")

        env = os.environ.copy()
        env["SSDB"] = address

        subprocess.run(
            [
                "foamSmartSimSvd",
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

        tensor_name = "fieldName_p-MPIrank_0"
        tensor = client.get_tensor(tensor_name)

        assert tensor.shape == (400, 50)
        assert tensor.dtype.name == "float64"

        print(f"Tensor name: {tensor_name}")
        print(f"Tensor shape: {tensor.shape}")
        print(f"Tensor dtype: {tensor.dtype}")
        print(f"Tensor range: {tensor.min()} to {tensor.max()}")

    finally:
        exp.stop(db)
