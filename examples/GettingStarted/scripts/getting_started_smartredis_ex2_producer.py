import argparse
import numpy as np
from smartredis import Client

parser = argparse.ArgumentParser()
parser.add_argument("--redis-port", type=int, default=None)
args = parser.parse_args()

client = Client(logger_name="producer")

tensor = np.random.rand(1, 1, 3, 3).astype(np.float64)
client.put_tensor(name="tensor", data=tensor)