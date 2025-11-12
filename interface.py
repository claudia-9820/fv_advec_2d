#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk
from fv2d import Grid2D, AdvectionSolver2D
from time_integrators import euler_step, rk2_heun_step
from initial_conditions import gaussian
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

def uniform_velocity(u0=1.0, v0=0.0):
    def vel(X, Y):
        return np.full_like(X, u0), np.full_like(X, v0)
    return vel

class FVApp:
    def __init__(self, root):
        self.root = root
        root.title("Simulation d'advection 2D (Volumes Finis)")
        root.geometry("900x700")

        # --- paramètres ---
        frame_params = ttk.LabelFrame(root, text="Paramètres de simulation")
        frame_params.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # champs
        self.var_nx = tk.IntVar(value=80)
        self.var_ny = tk.IntVar(value=80)
        self.var_u = tk.DoubleVar(value=1.0)
        self.var_v = tk.DoubleVar(value=0.0)
        self.var_tf = tk.DoubleVar(value=1.0)
        self.var_cfl = tk.DoubleVar(value=0.4)
        self.var_method = tk.StringVar(value="rk2")
        self.var_recon = tk.StringVar(value="linear")

        # construction du formulaire
        form = [
            ("Nx", self.var_nx),
            ("Ny", self.var_ny),
            ("Vitesse U", self.var_u),
            ("Vitesse V", self.var_v),
            ("Temps final", self.var_tf),
            ("CFL", self.var_cfl),
        ]
        for i, (label, var) in enumerate(form):
            ttk.Label(frame_params, text=label).grid(row=0, column=i*2, padx=3, pady=5)
            ttk.Entry(frame_params, textvariable=var, width=6).grid(row=0, column=i*2+1, padx=3, pady=5)

        ttk.Label(frame_params, text="Méthode temporelle").grid(row=1, column=0, padx=3, pady=5)
        ttk.OptionMenu(frame_params, self.var_method, "rk2", "rk2", "euler").grid(row=1, column=1, padx=3)

        ttk.Label(frame_params, text="Reconstruction").grid(row=1, column=2, padx=3, pady=5)
        ttk.OptionMenu(frame_params, self.var_recon, "linear", "linear", "constant").grid(row=1, column=3, padx=3)

        ttk.Button(frame_params, text="Lancer la simulation", command=self.run_simulation).grid(row=1, column=6, padx=10, pady=5)

        # --- graphique ---
        frame_plot = ttk.LabelFrame(root, text="Résultat de la simulation")
        frame_plot.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(6,5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_plot)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status = tk.StringVar(value="Prêt.")
        ttk.Label(root, textvariable=self.status).pack(side=tk.BOTTOM, pady=5)

    def run_simulation(self):
        nx = self.var_nx.get()
        ny = self.var_ny.get()
        u0 = self.var_u.get()
        v0 = self.var_v.get()
        final_time = self.var_tf.get()
        cfl = self.var_cfl.get()
        method = self.var_method.get()
        recon = self.var_recon.get()

        grid = Grid2D(nx, ny, 1.0, 1.0)
        vel = uniform_velocity(u0, v0)
        solver = AdvectionSolver2D(grid, vel, recon=recon)
        q = gaussian(grid, x0=0.25, y0=0.5, sigma=0.07)

        dt = solver.cfl_dt(cfl)
        t = 0.0
        start = time.time()

        self.ax.clear()
        im = self.ax.imshow(q, extent=[0,1,0,1], origin="lower", cmap="viridis", vmin=0, vmax=1)
        self.ax.set_title("Évolution du champ advecté")
        self.canvas.draw()

        while t < final_time - 1e-12:
            if t + dt > final_time:
                dt = final_time - t
            if method == "euler":
                q = euler_step(solver, q, dt)
            else:
                q = rk2_heun_step(solver, q, dt)
            t += dt
            if int(t/dt) % 3 == 0:  # rafraîchir tous les 3 pas
                im.set_data(q)
                self.canvas.draw()
                self.root.update_idletasks()

        end = time.time()
        self.status.set(f"Simulation terminée en {end-start:.2f}s (t_final={t:.2f})")
        im.set_data(q)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = FVApp(root)
    root.mainloop()
