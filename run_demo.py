import numpy as np
import matplotlib.pyplot as plt
from fv2d import Grid2D, AdvectionSolver2D
from time_integrators import euler_step, rk2_heun_step
from initial_conditions import gaussian

def uniform_velocity(u0=1.0, v0=0.5):
    def vel(X, Y):
        ny, nx = X.shape
        return np.full_like(X, u0), np.full_like(X, v0)
    return vel

def rotation_velocity(omega=2*np.pi):
    # solid body rotation around center (0.5,0.5)
    def vel(X, Y):
        cx, cy = 0.5, 0.5
        u = -omega * (Y - cy)
        v =  omega * (X - cx)
        return u, v
    return vel

def run_case(nx=100, ny=100, final_time=1.0, cfl=0.4, recon='linear', integrator='rk2'):
    grid = Grid2D(nx, ny, Lx=1.0, Ly=1.0)
    vel = uniform_velocity(u0=1.0, v0=0.0)  # pure x-translation
    solver = AdvectionSolver2D(grid, vel, recon=recon)
    q = gaussian(grid, x0=0.25, y0=0.5, sigma=0.05)
    dt = solver.cfl_dt(cfl_number=cfl)
    t = 0.0
    step = 0
    while t < final_time - 1e-12:
        if t + dt > final_time:
            dt = final_time - t
        if integrator == 'euler':
            q = euler_step(solver, q, dt)
        else:
            q = rk2_heun_step(solver, q, dt)
        t += dt
        step += 1
    return grid, q

if __name__ == '__main__':
    grid, q = run_case(nx=160, ny=160, final_time=1.0, recon='linear', integrator='rk2', cfl=0.45)
    plt.figure(figsize=(6,5))
    plt.contourf(grid.X, grid.Y, q, levels=50)
    plt.title("Solution at final time")
    plt.xlabel("x"); plt.ylabel("y")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("demo_solution.png", dpi=150)
    print("Saved demo_solution.png")