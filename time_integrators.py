import numpy as np

def euler_step(solver, q, dt):
    rhs = solver.compute_rhs(q)
    return q + dt * rhs

def rk2_heun_step(solver, q, dt):
    k1 = solver.compute_rhs(q)
    q1 = q + dt * k1
    k2 = solver.compute_rhs(q1)
    return q + 0.5 * dt * (k1 + k2)