# producer.py
from pathlib import Path
import argparse
import numpy as np
from sklearn.datasets import make_friedman1
from smartredis import Client, Dataset

# Initialise SmartRedis client
client = Client(logger_name="consumer")

# Retrieve the dataset created by producer.py
dataset = client.get_dataset("Friedman1_Dataset")

X = dataset.get_tensor("features").astype(np.float32)
y = dataset.get_tensor("targets").astype(np.float32)

# RedisAI model inputs must be standalone tensor keys
client.put_tensor("inference_features", X)

# Run backend inference
client.run_model(
    name="dnn",
    inputs=["inference_features"],
    outputs=["dnn_predictions"],
)

client.run_model(
    name="dnn_eqx",
    inputs=["inference_features"],
    outputs=["dnn_eqx_predictions"],
)

client.run_model(
    name="et",
    inputs=["inference_features"],
    outputs=["et_predictions"],
)

client.run_model(
    name="vr",
    inputs=["inference_features"],
    outputs=["vr_predictions"],
)

# Retrieve model outputs
dnn_predictions = client.get_tensor("dnn_predictions")
dnn_eqx_predictions = client.get_tensor("dnn_eqx_predictions")
et_predictions = client.get_tensor("et_predictions")
vr_predictions = client.get_tensor("vr_predictions")

# Store all inference results in one Dataset
result = Dataset("Friedman1_Inference")

result.add_tensor("targets", y)
result.add_tensor("dnn_predictions", dnn_predictions)
result.add_tensor("dnn_eqx_predictions", dnn_eqx_predictions)
result.add_tensor("et_predictions", et_predictions)
result.add_tensor("vr_predictions", vr_predictions)

result.add_meta_string("source_dataset", "Friedman1_Dataset")

client.put_dataset(result)

print("Backend inference completed successfully.")