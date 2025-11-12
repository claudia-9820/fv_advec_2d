import numpy as np

def periodic_idx(i, n):
    """Wrap index for periodic BC."""
    return i % n

def minmod(a, b):
    """Minmod limiter for arrays."""
    return np.where(np.sign(a) == np.sign(b), np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

class Grid2D:
    def __init__(self, nx, ny, Lx=1.0, Ly=1.0):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / nx
        self.dy = Ly / ny
        # cell centers coordinates
        x = (np.arange(nx) + 0.5) * self.dx
        y = (np.arange(ny) + 0.5) * self.dy
        self.X, self.Y = np.meshgrid(x, y, indexing='xy')  # shape (ny, nx)
        # note: array shape will be (ny, nx) to match meshgrid indexing

class AdvectionSolver2D:
    def __init__(self, grid: Grid2D, velocity_fn, recon='constant', limiter='minmod'):
        """
        velocity_fn: function(X, Y) -> (u, v) arrays shape (ny, nx)
        recon: 'constant' or 'linear'
        limiter: 'minmod' or None
        """
        self.grid = grid
        self.u, self.v = velocity_fn(grid.X, grid.Y)
        self.recon = recon
        self.limiter = limiter

    def apply_periodic(self, arr):
        # arr shape (ny, nx)
        return np.vstack([
            arr[-1:, :],
            arr,
            arr[:1, :]
        ])

    def compute_face_velocities(self):
        """Compute face-normal velocities at faces (averaging center values). 
        Returns u_face_x (ny, nx+1) for vertical faces normal x, and v_face_y (ny+1, nx) for horizontal faces normal y.
        """
        ny, nx = self.grid.ny, self.grid.nx
        # u on vertical faces located between cell j-1 and j (faces in x)
        u_face = np.zeros((ny, nx + 1))
        # average neighbouring cell-centered u
        u_padded = np.hstack([self.u[:, -1:], self.u, self.u[:, :1]])  # periodic in x
        for i in range(nx + 1):
            u_face[:, i] = 0.5 * (u_padded[:, i] + u_padded[:, i+1])
        # v on horizontal faces located between cell i-1 and i (faces in y)
        v_face = np.zeros((ny + 1, nx))
        v_padded = np.vstack([self.v[-1:, :], self.v, self.v[:1, :]])  # periodic in y
        for j in range(ny + 1):
            v_face[j, :] = 0.5 * (v_padded[j, :] + v_padded[j+1, :])
        return u_face, v_face

    def reconstruct(self, q):
        """Return left/right states at faces using reconstruction type."""
        ny, nx = self.grid.ny, self.grid.nx
        if self.recon == 'constant':
            # left state = cell value, right state = neighbor cell value -> flux upwinding will choose
            qL_x = np.hstack([q[:, :1], q])  # left state for faces (ny, nx+1) at face i is value of cell i-1
            qR_x = np.hstack([q, q[:, :1]])  # right state for faces (ny, nx+1) is value of cell i
            qL_y = np.vstack([q[:1, :], q])  # for horizontal faces (ny+1, nx)
            qR_y = np.vstack([q, q[:1, :]])
            return qL_x, qR_x, qL_y, qR_y
        elif self.recon == 'linear':
            # compute slopes with central difference and limiter
            qx_plus = np.hstack([q[:,1:], q[:, :1]])
            qx_minus = np.hstack([q[:, -1:], q[:, :-1]])
            dqdx = (qx_plus - qx_minus) / (2.0 * self.grid.dx)
            if self.limiter == 'minmod':
                # compute one-sided slopes
                s_plus = (qx_plus - q) / self.grid.dx
                s_minus = (q - qx_minus) / self.grid.dx
                slope = minmod(s_minus, s_plus)
            else:
                slope = dqdx
            # left/right states at vertical faces
            # face between cell i-1 and i: left = q_{i-1} + 0.5*s_{i-1}*dx ; right = q_i - 0.5*s_i*dx
            q_padded = np.hstack([q[:, -1:], q, q[:, :1]])
            slope_padded = np.hstack([slope[:, -1:], slope, slope[:, :1]])
            qL_x = np.zeros((ny, nx+1))
            qR_x = np.zeros((ny, nx+1))
            for i in range(nx+1):
                qL_x[:, i] = q_padded[:, i] + 0.5 * slope_padded[:, i] * self.grid.dx
                qR_x[:, i] = q_padded[:, i+1] - 0.5 * slope_padded[:, i+1] * self.grid.dx
            # same in y
            qy_plus = np.vstack([q[1:, :], q[:1, :]])
            qy_minus = np.vstack([q[-1:, :], q[:-1, :]])
            s_plus_y = (qy_plus - q) / self.grid.dy
            s_minus_y = (q - qy_minus) / self.grid.dy
            if self.limiter == 'minmod':
                slope_y = minmod(s_minus_y, s_plus_y)
            else:
                slope_y = (qy_plus - qy_minus) / (2.0 * self.grid.dy)
            q_padded_y = np.vstack([q[-1:, :], q, q[:1, :]])
            slope_padded_y = np.vstack([slope_y[-1:, :], slope_y, slope_y[:1, :]])
            qL_y = np.zeros((ny+1, nx))
            qR_y = np.zeros((ny+1, nx))
            for j in range(ny+1):
                qL_y[j, :] = q_padded_y[j, :] + 0.5 * slope_padded_y[j, :] * self.grid.dy
                qR_y[j, :] = q_padded_y[j+1, :] - 0.5 * slope_padded_y[j+1, :] * self.grid.dy
            return qL_x, qR_x, qL_y, qR_y
        else:
            raise ValueError("Unknown reconstruction")

    def compute_rhs(self, q):
        """Compute RHS of dq/dt = - div (u q, v q) using upwind on faces. q shape (ny,nx)"""
        ny, nx = self.grid.ny, self.grid.nx
        u_face, v_face = self.compute_face_velocities()
        # reconstruct states at faces
        qL_x, qR_x, qL_y, qR_y = self.reconstruct(q)
        # fluxes on vertical faces (x-normal): F = u_face * q_upwind
        # upwind: if u_face > 0 take right state of left cell (qR_x[:, i-1])? careful with indexing:
        # our qL_x[:, i] is state in cell i-1 extrapolated to face; qR_x[:, i] is state in cell i extrapolated to face.
        F_x = np.where(u_face >= 0, u_face * qL_x, u_face * qR_x)  # shape (ny, nx+1)
        # fluxes on horizontal faces (y-normal): G = v_face * q_upwind
        G_y = np.where(v_face >= 0, v_face * qL_y, v_face * qR_y)  # shape (ny+1, nx)
        # divergence: (F_x[:, i+1] - F_x[:, i]) / dx + (G_y[j+1,:] - G_y[j,:]) / dy
        div = (F_x[:, 1:] - F_x[:, :-1]) / self.grid.dx + (G_y[1:, :] - G_y[:-1, :]) / self.grid.dy
        return -div

    def cfl_dt(self, cfl_number=0.5):
        # local max vel
        maxu = np.max(np.abs(self.u))
        maxv = np.max(np.abs(self.v))
        if maxu == 0 and maxv == 0:
            return 1.0
        dt_x = self.grid.dx / (maxu + 1e-16)
        dt_y = self.grid.dy / (maxv + 1e-16)
        dt = cfl_number * min(dt_x, dt_y)
        return dt