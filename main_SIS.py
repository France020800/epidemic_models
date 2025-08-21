import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap


class SISImmunitySimulation:
    def __init__(self, master):
        self.master = master
        master.title("SIS Model with Partial Immunity")

        # Parameters
        self.size = 100
        self.density = 0.6
        self.infectivity = 0.3
        self.recovery_rate = 0.1
        self.jump_infectivity = 0.01
        self.partial_immunity_percentage = 0.5
        self.time = 0
        self.max_time = 1000

        # Simulation state
        self.running = False
        self.job = None
        self.step_delay = 100

        # Data for the logistic curve
        self.susceptible_count = []
        self.infected_count = []
        self.time_steps = []

        # Colormaps (0:Susceptible, 1:Infected, 2:Empty, 3:Partially Immune)
        self.cmap = ListedColormap(["green", "red", "white", "yellowgreen"])

        # Init grid
        self.reset_grid()

        # --- Matplotlib Figures ---
        self.fig_grid, self.ax_grid = plt.subplots(figsize=(4, 4))
        self.canvas_grid = FigureCanvasTkAgg(self.fig_grid, master)
        self.canvas_grid.get_tk_widget().grid(row=0, column=0, rowspan=12, padx=10, pady=10)

        self.fig_curve, self.ax_curve = plt.subplots(figsize=(4, 3))
        self.canvas_curve = FigureCanvasTkAgg(self.fig_curve, master)
        self.canvas_curve.get_tk_widget().grid(row=12, column=0, rowspan=2, padx=10, pady=10)

        # --- Sliders ---
        ttk.Label(master, text="Time").grid(row=0, column=1, sticky="w")
        self.time_slider = tk.Scale(master, from_=0, to=self.max_time,
                                    orient="horizontal", command=self.update_time, length=200)
        self.time_slider.grid(row=1, column=1, sticky="ew")

        ttk.Label(master, text="Population density").grid(row=2, column=1, sticky="w")
        self.density_slider = tk.Scale(master, from_=0.1, to=1.0, resolution=0.1,
                                       orient="horizontal", command=self.update_density, length=200)
        self.density_slider.set(self.density)
        self.density_slider.grid(row=3, column=1, sticky="ew")

        ttk.Label(master, text="Infectivity").grid(row=4, column=1, sticky="w")
        self.infectivity_slider = tk.Scale(master, from_=0.0, to=1.0, resolution=0.05,
                                           orient="horizontal", command=self.update_infectivity, length=200)
        self.infectivity_slider.set(self.infectivity)
        self.infectivity_slider.grid(row=5, column=1, sticky="ew")

        ttk.Label(master, text="Recovery Rate").grid(row=6, column=1, sticky="w")
        self.recovery_slider = tk.Scale(master, from_=0.0, to=1.0, resolution=0.05,
                                        orient="horizontal", command=self.update_recovery_rate, length=200)
        self.recovery_slider.set(self.recovery_rate)
        self.recovery_slider.grid(row=7, column=1, sticky="ew")

        ttk.Label(master, text="Jump Infectivity").grid(row=8, column=1, sticky="w")
        self.jump_slider = tk.Scale(master, from_=0.0, to=0.2, resolution=0.005,
                                    orient="horizontal", command=self.update_jump_infectivity, length=200)
        self.jump_slider.set(self.jump_infectivity)
        self.jump_slider.grid(row=9, column=1, sticky="ew")

        ttk.Label(master, text="Partial Immunity (%)").grid(row=10, column=1, sticky="w")
        self.partial_immunity_slider = tk.Scale(master, from_=0, to=100, resolution=5,
                                                orient="horizontal", command=self.update_partial_immunity, length=200)
        self.partial_immunity_slider.set(self.partial_immunity_percentage * 100)
        self.partial_immunity_slider.grid(row=11, column=1, sticky="ew")

        # --- Control Buttons ---
        self.control_button = ttk.Button(master, text="Play", command=self.toggle_simulation)
        self.control_button.grid(row=12, column=1, sticky="ew", pady=5)

        self.reset_button = ttk.Button(master, text="Reset Simulation", command=self.reset_simulation)
        self.reset_button.grid(row=13, column=1, sticky="ew", pady=5)

        # Initial draw must have a data point
        self.record_data()
        self.draw_grid()
        self.draw_logistic_curve()

    # --- Updates ---
    def update_time(self, val):
        t = int(val)
        if self.running: return

        if self.time < t:
            while self.time < t:
                self.step()
                self.time += 1
        elif self.time > t:
            self.reset_grid()
            for _ in range(t):
                self.step()
            self.time = t
        self.draw_grid()
        self.draw_logistic_curve()

    def update_density(self, val):
        self.density = float(val)
        self.reset_simulation()

    def update_infectivity(self, val):
        self.infectivity = float(val)

    def update_recovery_rate(self, val):
        self.recovery_rate = float(val)

    def update_jump_infectivity(self, val):
        self.jump_infectivity = float(val)

    def update_partial_immunity(self, val):
        self.partial_immunity_percentage = float(val) / 100.0

    # --- Simulation Logic ---
    def reset_grid(self):
        self.grid = np.full((self.size, self.size), -1)  # -1 = empty
        mask = np.random.rand(self.size, self.size) < self.density
        self.grid[mask] = 0  # 0 = susceptible
        if not np.any(self.grid == 0):
            self.grid[self.size // 2, self.size // 2] = 0
        initial_infected_pos = (self.size // 2, self.size // 2)
        if self.grid[initial_infected_pos] == -1:
            susceptible_indices = np.argwhere(self.grid == 0)
            if len(susceptible_indices) > 0:
                initial_infected_pos = susceptible_indices[np.random.randint(len(susceptible_indices))]
            else:
                self.grid[self.size // 2, self.size // 2] = 0
                initial_infected_pos = (self.size // 2, self.size // 2)
        self.grid[initial_infected_pos] = 1  # 1 = infected
        self.time = 0
        if hasattr(self, "time_slider"):
            self.time_slider.set(0)

    def reset_simulation(self):
        self.stop_simulation()
        self.susceptible_count = []
        self.infected_count = []
        self.time_steps = []
        self.reset_grid()
        self.record_data()
        self.draw_grid()
        self.draw_logistic_curve()

    def _get_non_neighbor_susceptible_cell(self, i, j, susceptible_indices):
        """Finds a random susceptible cell that is not adjacent to (i,j)."""
        if len(susceptible_indices) == 0:
            return None, None

        valid_targets = []
        for ni, nj in susceptible_indices:
            is_neighbor = abs(ni - i) <= 1 and abs(nj - j) <= 1
            if not is_neighbor:
                valid_targets.append((ni, nj))

        if len(valid_targets) > 0:
            return valid_targets[np.random.randint(len(valid_targets))]
        else:
            return None, None

    def step(self):
        new_grid = self.grid.copy()
        infected_indices = np.argwhere(self.grid == 1)
        susceptible_indices = np.argwhere(np.logical_or(self.grid == 0, self.grid == 3))

        # 1. Neighbor-based infection
        for i, j in infected_indices:
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    if new_grid[ni, nj] == 0 and np.random.rand() < self.infectivity:
                        new_grid[ni, nj] = 1
                    elif new_grid[ni, nj] == 3 and np.random.rand() < (
                            self.infectivity * self.partial_immunity_percentage):
                        new_grid[ni, nj] = 1

            # 2. Recovery from infected state
            if np.random.rand() < self.recovery_rate:
                new_grid[i, j] = 3

        # 3. "Jump" infection
        for i, j in infected_indices:
            if np.random.rand() < self.jump_infectivity:
                ti, tj = self._get_non_neighbor_susceptible_cell(i, j, susceptible_indices)
                if ti is not None:
                    if new_grid[ti, tj] == 0:
                        if np.random.rand() < self.infectivity:
                            new_grid[ti, tj] = 1
                    elif new_grid[ti, tj] == 3:
                        if np.random.rand() < (self.infectivity * self.partial_immunity_percentage):
                            new_grid[ti, tj] = 1

        self.grid = new_grid

    def record_data(self):
        """Records the current population counts for plotting."""
        self.time_steps.append(self.time)
        susceptible_total = np.sum(self.grid == 0) + np.sum(self.grid == 3)
        self.susceptible_count.append(susceptible_total)
        self.infected_count.append(np.sum(self.grid == 1))

    def draw_grid(self):
        self.ax_grid.clear()
        show_grid = np.ma.masked_where(self.grid == -1, self.grid)
        self.ax_grid.imshow(show_grid, cmap=self.cmap, vmin=0, vmax=3)
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        self.canvas_grid.draw_idle()

    def draw_logistic_curve(self):
        self.ax_curve.clear()
        self.ax_curve.plot(self.time_steps, self.susceptible_count, label='Susceptible', color='green')
        self.ax_curve.plot(self.time_steps, self.infected_count, label='Infected', color='red')
        self.ax_curve.set_title("SIS Model with Partial Immunity")
        self.ax_curve.set_xlabel("Time Step")
        self.ax_curve.set_ylabel("Population")
        self.ax_curve.legend()
        self.ax_curve.grid(True)
        self.canvas_curve.draw_idle()

    # --- Play/Pause Functionality ---
    def toggle_simulation(self):
        if self.running:
            self.stop_simulation()
        else:
            self.start_simulation()

    def start_simulation(self):
        if not self.running and self.time < self.max_time:
            self.running = True
            self.control_button.config(text="Pause")
            self.run_step()

    def stop_simulation(self):
        if self.running:
            self.running = False
            self.control_button.config(text="Play")
            if self.job is not None:
                self.master.after_cancel(self.job)
                self.job = None

    def run_step(self):
        infected_count = np.sum(self.grid == 1)
        if infected_count == 0:
            print("Simulation stopped: No more infected people remain.")
            self.stop_simulation()
            self.draw_logistic_curve()
            return
        if self.running and self.time < self.max_time:
            infected_count = np.sum(self.grid == 1)

            self.step()
            self.time += 1
            self.time_slider.set(self.time)

            self.record_data()
            self.draw_grid()
            self.draw_logistic_curve()

            self.job = self.master.after(self.step_delay, self.run_step)
        else:
            self.stop_simulation()
            self.draw_logistic_curve()


if __name__ == "__main__":
    root = tk.Tk()
    app = SISImmunitySimulation(root)
    root.mainloop()