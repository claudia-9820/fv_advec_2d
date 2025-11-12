import numpy as np

def gaussian(grid, x0=0.5, y0=0.5, sigma=0.05):
    X, Y = grid.X, grid.Y
    return np.exp(-(((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)))

def square_plate(grid, x0=0.5, y0=0.5, half_width=0.1):
    X, Y = grid.X, grid.Y
    return np.where((np.abs(X-x0) <= half_width) & (np.abs(Y-y0) <= half_width), 1.0, 0.0)