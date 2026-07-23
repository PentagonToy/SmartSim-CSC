import argparse
import os
from smartredis import Client

parser = argparse.ArgumentParser()
parser.add_argument("--redis-port", type=int, default=None)
args = parser.parse_args()

client = Client(logger_name="consumer")

sources = os.environ.get("SSKEYIN", "")
sources = [s.strip() for s in sources.split(",") if s.strip()]

for src in sources:
    client.set_data_source(src)
    client.poll_tensor(name="tensor", poll_frequency_ms=200, num_tries=10000)
    tensor = client.get_tensor(name="tensor")
    print(f"Tensor for {src} is: {tensor}")