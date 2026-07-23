# producer_ldc.py
from pathlib import Path
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from smartredis import Client, Dataset

# SmartSim / SmartRedis settings
# Parsing command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--Re", type=float, default=400.0, help="Reynolds number")
parser.add_argument("--nx", type=int, default=64, help="Cell number of x-axis")
parser.add_argument("--ny", type=int, default=64, help="Cell number of y-axis")
parser.add_argument("--nt", type=int, default=5000, help="Time steps")
parser.add_argument("--endTime", type=float, default=10.0, help="End time")
parser.add_argument("--save_interval", type=int, default=100, help="Interval for saving vorticity data")
args = parser.parse_args()

# Parsed arguments (configuration)
SEED = args.seed
Re = args.Re
nx, ny = args.nx, args.ny
nt = args.nt
endTime = args.endTime
save_interval = args.save_interval
key = jax.random.PRNGKey(args.seed)

# Initialise SmartRedis client
client = Client(logger_name="producer")

#  LDC Solver setup
def make_solver(nx, ny, nt, endTime, Re):
    lx, ly = 1.0, 1.0
    dx, dy = lx / nx, ly / ny
    nu = 1.0 / Re
    U_lid = 1.0
    dt = endTime / nt

    # --- Staggered grid ---
    u0 = jnp.zeros((ny, nx + 1))
    v0 = jnp.zeros((ny + 1, nx))
    p0 = jnp.zeros((ny, nx))

    # Ghost-cell helpers for tangential no-slip BCs
    def extend_u_y(u):
        ghost_bot = -u[0:1, :]
        ghost_top = 2.0 * U_lid - u[-1:, :]
        return jnp.concatenate([ghost_bot, u, ghost_top], axis=0)

    def extend_v_x(v):
        ghost_left  = -v[:, 0:1]
        ghost_right = -v[:, -1:]
        return jnp.concatenate([ghost_left, v, ghost_right], axis=1)

    # Pressure Poisson solver (Jacobi, Neumann BCs)
    def pressure_poisson(p, rhs):
        dx2 = dx * dx
        def body_fn(_, p_val):
            p_pad = jnp.pad(p_val, 1, mode='edge')
            p_new = 0.25 * (
                p_pad[1:-1, 2:]  + p_pad[1:-1, :-2] +
                p_pad[2:, 1:-1]  + p_pad[:-2, 1:-1] -
                dx2 * rhs
            )
            return p_new
        p = jax.lax.fori_loop(0, 500, body_fn, p)
        return p - jnp.mean(p)

    # Single time step
    def step_fn(carry, _):
        u, v, p = carry

        # u-momentum
        u_ext = extend_u_y(u)

        diff_u = (
            nu * (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / dx**2 +
            nu * (u_ext[2:, 1:-1] - 2.0 * u_ext[1:-1, 1:-1]
                  + u_ext[:-2, 1:-1]) / dy**2
        )

        du_dx = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)
        du_dy = (u_ext[2:, 1:-1] - u_ext[:-2, 1:-1]) / (2.0 * dy)
        v_at_u = 0.25 * (v[:-1, :-1] + v[:-1, 1:]
                       + v[1:,  :-1] + v[1:,  1:])
        adv_u = u[:, 1:-1] * du_dx + v_at_u * du_dy

        u_star = u.at[:, 1:-1].set(u[:, 1:-1] + dt * (diff_u - adv_u))
        u_star = u_star.at[:, 0].set(0.0)
        u_star = u_star.at[:, -1].set(0.0)

        # v-momentum
        v_ext = extend_v_x(v)

        diff_v = (
            nu * (v_ext[1:-1, 2:] - 2.0 * v_ext[1:-1, 1:-1]
                  + v_ext[1:-1, :-2]) / dx**2 +
            nu * (v[2:, :] - 2.0 * v[1:-1, :] + v[:-2, :]) / dy**2
        )

        dv_dx = (v_ext[1:-1, 2:] - v_ext[1:-1, :-2]) / (2.0 * dx)
        dv_dy = (v[2:, :] - v[:-2, :]) / (2.0 * dy)
        u_at_v = 0.25 * (u[:-1, :-1] + u[:-1, 1:]
                       + u[1:,  :-1] + u[1:,  1:])
        adv_v = u_at_v * dv_dx + v[1:-1, :] * dv_dy

        v_star = v.at[1:-1, :].set(v[1:-1, :] + dt * (diff_v - adv_v))
        v_star = v_star.at[0, :].set(0.0)
        v_star = v_star.at[-1, :].set(0.0)

        # --- Pressure Poisson ---
        div = ((u_star[:, 1:] - u_star[:, :-1]) / dx +
               (v_star[1:, :] - v_star[:-1, :]) / dy)
        p = pressure_poisson(p, div / dt)

        # --- Projection ---
        u_new = u_star.at[:, 1:-1].add(-dt / dx * (p[:, 1:] - p[:, :-1]))
        v_new = v_star.at[1:-1, :].add(-dt / dy * (p[1:, :] - p[:-1, :]))

        u_new = u_new.at[:, 0].set(0.0)
        u_new = u_new.at[:, -1].set(0.0)
        v_new = v_new.at[0, :].set(0.0)
        v_new = v_new.at[-1, :].set(0.0)

        return (u_new, v_new, p), None

    return (u0, v0, p0), step_fn

# Build the solver
init_state, step_fn = make_solver(nx, ny, nt, endTime, Re)

# Time integration with periodic export
dt = endTime / nt
chunk = save_interval
state = init_state

print(f"Starting simulation with nx={nx}, ny={ny}, nt={nt}, Re={Re}, dt={dt}, save_interval={save_interval}")

for step in range(0, nt, chunk):
    # Run the solver for 'chunk' time steps
    actual_chunk = min(chunk, nt - step)
    state, _ = jax.lax.scan(step_fn, state, None, length=actual_chunk)
    
    current_step = step + actual_chunk
    current_time = current_step * dt
    
    # Transfer to CPU and convert to numpy
    u_np, v_np, p_np = map(np.asarray, jax.device_get(state))
    
    # Compute cell-centred velocity magnitude
    u_c = 0.5 * (u_np[:, 1:] + u_np[:, :-1])
    v_c = 0.5 * (v_np[1:, :] + v_np[:-1, :])
    speed = np.sqrt(u_c**2 + v_c**2)
    
    # Pack into a SmartRedis dataset
    dataset = Dataset(f"data_{current_step}")
    dataset.add_tensor("u", u_np)
    dataset.add_tensor("v", v_np)
    dataset.add_tensor("p", p_np)
    dataset.add_tensor("speed", speed)
    dataset.add_tensor("u_centre", u_c)
    dataset.add_tensor("v_centre", v_c)
    
    # Metadata
    dataset.add_meta_scalar("time_step", current_step)
    dataset.add_meta_scalar("sim_time", current_time)
    dataset.add_meta_scalar("Re", Re)
    dataset.add_meta_scalar("nx", nx)
    dataset.add_meta_scalar("ny", ny)
    
    client.put_dataset(dataset)
    print(f"Saved dataset for time step {current_step} at simulation time {current_time:.4f}")

print(f"Simulation completed for {nt} time steps. Data pushed to SmartRedis.")