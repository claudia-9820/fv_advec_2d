import numpy as np
from fv2d import Grid2D, AdvectionSolver2D
from time_integrators import rk2_heun_step
from initial_conditions import gaussian
import math

def uniform_velocity(u0=1.0, v0=0.0):
    def vel(X, Y):
        return np.full_like(X, u0), np.full_like(X, v0)
    return vel

def l2_error(q_num, q_ref, grid):
    diff = q_num - q_ref
    return math.sqrt(np.sum(diff**2) * grid.dx * grid.dy)

def reference_solution(grid, q0, vel, final_time):
    # for uniform velocity analytic shift; only for uniform vel
    u = float(vel(grid.X, grid.Y)[0][0])
    v = float(vel(grid.X, grid.Y)[1][0])
    # shift positions by u*final_time, v*final_time (mod periodic)
    Xs = (grid.X - u*final_time) % grid.Lx
    Ys = (grid.Y - v*final_time) % grid.Ly
    # interpolate q0 defined on grid centers -> we approximate by nearest neighbor (since grid is same)
    # find original index that maps to these coordinates
    xi = (Xs / grid.dx - 0.5).astype(int) % grid.nx
    yi = (Ys / grid.dy - 0.5).astype(int) % grid.ny
    return q0[yi, xi]

def run_convergence():
    ns = [32, 64, 128, 256]
    errors = []
    final_time = 1.0
    for n in ns:
        grid = Grid2D(n, n, 1.0, 1.0)
        vel = uniform_velocity(u0=1.0, v0=0.0)
        solver = AdvectionSolver2D(grid, vel, recon='linear')
        q0 = gaussian(grid, x0=0.25, y0=0.5, sigma=0.05)
        q = q0.copy()
        dt = solver.cfl_dt(0.4)
        t = 0.0
        while t < final_time - 1e-12:
            if t + dt > final_time:
                dt = final_time - t
            q = rk2_heun_step(solver, q, dt)
            t += dt
        qref = reference_solution(grid, q0, vel, final_time)
        err = l2_error(q, qref, grid)
        errors.append(err)
        print(f"n={n}, L2 error={err:.3e}")
    # compute observed order
    for i in range(1, len(ns)):
        order = np.log(errors[i-1]/errors[i]) / np.log(ns[i]/ns[i-1])
        print(f"order between {ns[i-1]} and {ns[i]}: {order:.2f}")

if __name__ == '__main__':
    run_convergence()