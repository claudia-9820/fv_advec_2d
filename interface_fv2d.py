#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # backend pour Tkinter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ============================================================
# Schéma simple d'advection 2D (Euler explicite)
# ============================================================
def advect(u, v, C, dx, dy, dt):
    dCdx = (np.roll(C, -1, axis=1) - np.roll(C, 1, axis=1)) / (2 * dx)
    dCdy = (np.roll(C, -1, axis=0) - np.roll(C, 1, axis=0)) / (2 * dy)
    return C - dt * (u * dCdx + v * dCdy)

def init_gaussian(nx, ny, Lx, Ly, x0, y0, sigma):
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    return np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

# ============================================================
# Classe interface
# ============================================================
class AdvectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Simulation d’advection 2D - Méthode des volumes finis")
        master.geometry("950x600")

        # ==== PARAMÈTRES ====
        self.params_frame = ttk.LabelFrame(master, text="Paramètres")
        self.params_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.entries = {}
        params = {
            "nx": 100,
            "ny": 100,
            "Lx": 1.0,
            "Ly": 1.0,
            "u": 1.0,
            "v": 0.5,
            "dt": 0.002,
            "tmax": 2.0,
            "sigma": 0.1,
        }

        for i, (name, val) in enumerate(params.items()):
            ttk.Label(self.params_frame, text=name).grid(row=i, column=0, sticky="w", pady=2)
            entry = ttk.Entry(self.params_frame, width=10)
            entry.insert(0, str(val))
            entry.grid(row=i, column=1, pady=2)
            self.entries[name] = entry

        # ==== BOUTONS ====
        self.btn_frame = ttk.Frame(self.params_frame)
        self.btn_frame.grid(row=len(params), columnspan=2, pady=10)

        self.start_btn = ttk.Button(self.btn_frame, text="▶ Démarrer", command=self.start)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.pause_btn = ttk.Button(self.btn_frame, text="⏸ Pause", command=self.pause)
        self.pause_btn.grid(row=0, column=1, padx=5)

        self.reset_btn = ttk.Button(self.btn_frame, text="🔁 Réinitialiser", command=self.reset)
        self.reset_btn.grid(row=0, column=2, padx=5)

        # ==== FIGURE MATPLOTLIB ====
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.anim = None
        self.running = False

    # ----------------------------------------------------------
    def start(self):
        if self.anim is not None:
            self.running = True
            return  # si déjà lancé

        # Lecture des paramètres
        self.nx = int(self.entries["nx"].get())
        self.ny = int(self.entries["ny"].get())
        self.Lx = float(self.entries["Lx"].get())
        self.Ly = float(self.entries["Ly"].get())
        self.u = float(self.entries["u"].get())
        self.v = float(self.entries["v"].get())
        self.dt = float(self.entries["dt"].get())
        self.tmax = float(self.entries["tmax"].get())
        self.sigma = float(self.entries["sigma"].get())

        # Initialisation du champ
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.C = init_gaussian(self.nx, self.ny, self.Lx, self.Ly, 0.3, 0.3, self.sigma)

        self.t = 0.0
        self.running = True

        self.ax.clear()
        self.img = self.ax.imshow(self.C, extent=[0, self.Lx, 0, self.Ly], origin="lower",
                                  cmap="viridis", vmin=0, vmax=1)
        self.ax.set_title("Évolution du champ advecté")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        self.anim = FuncAnimation(self.fig, self.update, interval=30, blit=False)
        self.canvas.draw()

    # ----------------------------------------------------------
    def update(self, frame):
        if not self.running:
            return self.img,
        if self.t >= self.tmax:
            self.running = False
            return self.img,

        self.C = advect(self.u, self.v, self.C, self.dx, self.dy, self.dt)
        self.t += self.dt
        self.img.set_data(self.C)
        self.ax.set_title(f"t = {self.t:.2f} s")
        return self.img,

    # ----------------------------------------------------------
    def pause(self):
        self.running = not self.running
        self.pause_btn.config(text="▶ Reprendre" if not self.running else "⏸ Pause")

    # ----------------------------------------------------------
    def reset(self):
        if self.anim:
            self.anim.event_source.stop()
        self.anim = None
        self.ax.clear()
        self.canvas.draw()
        self.running = False

# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AdvectionApp(root)
    root.mainloop()
