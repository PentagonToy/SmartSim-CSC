# producer_vk.py
from pathlib import Path
import argparse
import numpy as np
from smartredis import Client, Dataset

# SmartSim / SmartRedis settings
# Parsing command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--x_res", type=int, default=400, help="Cell number of x-axis")
parser.add_argument("--y_res", type=int, default=100, help="Cell number of y-axis")
parser.add_argument("--tau", type=float, default=0.55, help="Relaxation time")
parser.add_argument("--steps", type=int, default=1000, help="Number of time steps")
parser.add_argument("--save_interval", type=int, default=5, help="Interval for saving vorticity data")
args = parser.parse_args()

# Parsed arguments (configuration)
SEED = args.seed
X_RES, Y_RES = args.x_res, args.y_res
TAU = args.tau
STEPS = args.steps
SAVE_INTERVAL = args.save_interval
np.random.seed(SEED)

# Initialise SmartRedis client
client = Client(logger_name="producer")

# LBM Initialisation
x_res, y_res = X_RES, Y_RES
tau = TAU

# Velocity directions
idxs = np.arange(9)
cxs = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
cys = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
weights = np.array([4/9] + [1/9] * 4 + [1/36] * 4)

# Grid
X, Y = np.meshgrid(np.arange(x_res), np.arange(y_res))

# Initialise distribution function to equilibrium with mean horizontal flow
u0 = 0.1
rho0 = 1.0

F = np.zeros((y_res, x_res, 9))
for i, cx, cy, w in zip(idxs, cxs, cys, weights):
    cu = cx * u0
    F[:, :, i] = rho0 * w * (1 + 3 * cu + 9 * cu**2 / 2 - 3 * u0**2 / 2)

# Add small perturbation to break symmetry
F += 0.01 * np.random.randn(y_res, x_res, 9)

# Save cylinder location to database
cylinder = (X - x_res / 4) ** 2 + (Y - y_res / 2) ** 2 < (y_res / 4) ** 2
client.put_tensor("cylinder", cylinder.astype(np.int8))

for time_step in range(STEPS):

    # Streaming
    for i, cx, cy in zip(idxs, cxs, cys):
        F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
        F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

    # Bounce-back: save boundary populations and reverse directions
    bndryF = F[cylinder, :]
    bndryF = bndryF[:, [0, 3, 4, 1, 2, 7, 8, 5, 6]]

    # Macroscopic quantities
    rho = np.sum(F, axis=2)
    ux = np.sum(F * cxs, axis=2) / rho
    uy = np.sum(F * cys, axis=2) / rho

    # Equilibrium distribution
    Feq = np.zeros(F.shape)
    for i, cx, cy, w in zip(idxs, cxs, cys, weights):
        cu = cx * ux + cy * uy
        Feq[:, :, i] = rho * w * (
            1
            + 3 * cu
            + 9 * cu ** 2 / 2
            - 3 * (ux ** 2 + uy ** 2) / 2
        )

    # Collision
    F += -(1.0 / tau) * (F - Feq)

    # Apply bounce-back
    F[cylinder, :] = bndryF

    # Create a SmartRedis dataset with vorticity data
    if time_step % SAVE_INTERVAL == 0:
        dataset = Dataset(f"data_{time_step}")
        dataset.add_tensor("ux", ux)
        dataset.add_tensor("uy", uy)

        # Put Dataset in db
        client.put_dataset(dataset)
        
print(f"Simulation completed for {STEPS} time steps. Vorticity data pushed to SmartRedis.")