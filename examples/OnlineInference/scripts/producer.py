# producer.py
from pathlib import Path
import argparse
import numpy as np
from sklearn.datasets import make_friedman1
from smartredis import Client, Dataset

# SmartSim / SmartRedis settings
# Parsing command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--n_samples", type=int, default=int(1e6), help="Number of samples")
parser.add_argument("--sigma_noise", type=float, default=0.5, help="Noise level")
args = parser.parse_args()

# Parsed arguments (configuration)
SEED = args.seed
N_SAMPLES = args.n_samples
SIGMA_NOISE = args.sigma_noise
np.random.seed(SEED)

# Initialise SmartRedis client
client = Client(logger_name="producer")

# Generate synthetic dataset using Friedman #1 function
X, y = make_friedman1(n_samples=N_SAMPLES, n_features=10, noise=SIGMA_NOISE, random_state=SEED)

print(f"Generated synthetic dataset with {N_SAMPLES} samples and 10 features.")

# Pack and push the dataset to SmartRedis
dataset = Dataset("Friedman1_Dataset")
dataset.add_tensor("features", X)
dataset.add_tensor("targets", y)
client.put_dataset(dataset)
print("Dataset pushed to SmartRedis successfully.")