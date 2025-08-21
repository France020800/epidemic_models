import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib

# Set the backend for Matplotlib to work with Tkinter
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import time


class SIRS_Movement_Simulation:
    """
    A GUI application to run and visualize a SIRS epidemic model with mobile agents.
    Contagion occurs when a susceptible agent meets an infected agent on the grid.
    Includes a fixed, initially vaccinated population with adjustable effectiveness and a mortality rate.
    """

    def __init__(self, master):
        self.master = master
        master.title("SIRS Movement Simulation")

        # --- Simulation Parameters ---
        self.size = 100
        self.agent_count = 500
        self.infectivity = 0.3
        self.recovery_rate = 0.1
        self.waning_immunity_rate = 0.01
        self.jump_infectivity = 0.001
        self.vaccination_rate = 0.5  # Percentage of population initially vaccinated
        self.vaccine_effectiveness = 0.8  # Reduction in infection chance (0.0 to 1.0)
        self.mortality_rate = 0.01  # Percentage of infected agents who are removed per step
        self.time = 0
        self.max_time = 1000

        # --- Total agents removed from the simulation ---
        self.total_removed = 0

        # --- Simulation State ---
        self.running = False
        self.job = None
        self.step_delay = 50
        self.agents = []

        # --- Data for the Logistic Curve ---
        self.susceptible_count = []
        self.infected_count = []
        self.recovered_count = []
        self.vaccinated_count = []
        self.mortality_count = []
        self.time_steps = []

        # --- GUI Elements ---
        self.main_frame = ttk.Frame(master, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- Matplotlib Figures ---
        self.fig_grid, self.ax_grid = plt.subplots(figsize=(5, 5))
        self.canvas_grid = FigureCanvasTkAgg(self.fig_grid, self.main_frame)
        self.canvas_grid.get_tk_widget().grid(row=0, column=0, rowspan=17, padx=10, pady=10)

        self.fig_curve, self.ax_curve = plt.subplots(figsize=(5, 3))
        self.canvas_curve = FigureCanvasTkAgg(self.fig_curve, self.main_frame)
        self.canvas_curve.get_tk_widget().grid(row=17, column=0, rowspan=2, padx=10, pady=10)

        # --- Sliders and Labels for Parameters ---
        self.create_sliders()

        # --- Control Buttons ---
        self.control_button = ttk.Button(self.main_frame, text="Play", command=self.toggle_simulation)
        self.control_button.grid(row=16, column=1, sticky="ew", pady=5)

        self.reset_button = ttk.Button(self.main_frame, text="Reset Simulation", command=self.reset_simulation)
        self.reset_button.grid(row=17, column=1, sticky="ew", pady=5)

        # Initialize simulation
        self.reset_simulation()
        self.draw_plot()

    def create_sliders(self):
        """Creates the sliders and labels for all simulation parameters."""
        ttk.Label(self.main_frame, text="Infectivity").grid(row=0, column=1, sticky="w")
        self.infectivity_slider = tk.Scale(self.main_frame, from_=0.0, to=1.0, resolution=0.01,
                                           orient="horizontal", command=self.update_infectivity, length=200)
        self.infectivity_slider.set(self.infectivity)
        self.infectivity_slider.grid(row=1, column=1, sticky="ew")

        ttk.Label(self.main_frame, text="Recovery Rate").grid(row=2, column=1, sticky="w")
        self.recovery_slider = tk.Scale(self.main_frame, from_=0.0, to=1.0, resolution=0.01,
                                        orient="horizontal", command=self.update_recovery_rate, length=200)
        self.recovery_slider.set(self.recovery_rate)
        self.recovery_slider.grid(row=3, column=1, sticky="ew")

        ttk.Label(self.main_frame, text="Waning Immunity").grid(row=4, column=1, sticky="w")
        self.waning_immunity_slider = tk.Scale(self.main_frame, from_=0.0, to=0.2, resolution=0.001,
                                               orient="horizontal", command=self.update_waning_immunity_rate,
                                               length=200)
        self.waning_immunity_slider.set(self.waning_immunity_rate)
        self.waning_immunity_slider.grid(row=5, column=1, sticky="ew")

        ttk.Label(self.main_frame, text="Jump Infectivity").grid(row=6, column=1, sticky="w")
        self.jump_slider = tk.Scale(self.main_frame, from_=0.0, to=0.01, resolution=0.0001,
                                    orient="horizontal", command=self.update_jump_infectivity, length=200)
        self.jump_slider.set(self.jump_infectivity)
        self.jump_slider.grid(row=7, column=1, sticky="ew")

        ttk.Label(self.main_frame, text="Number of Agents").grid(row=8, column=1, sticky="w")
        self.agent_slider = tk.Scale(self.main_frame, from_=100, to=10000, resolution=10,
                                     orient="horizontal", command=self.update_agent_count, length=200)
        self.agent_slider.set(self.agent_count)
        self.agent_slider.grid(row=9, column=1, sticky="ew")

        ttk.Label(self.main_frame, text="Vaccination Rate (%)").grid(row=10, column=1, sticky="w")
        self.vaccination_slider = tk.Scale(self.main_frame, from_=0, to=100, resolution=1,
                                           orient="horizontal", command=self.update_vaccination_rate, length=200)
        self.vaccination_slider.set(self.vaccination_rate * 100)
        self.vaccination_slider.grid(row=11, column=1, sticky="ew")

        ttk.Label(self.main_frame, text="Vaccine Effectiveness (%)").grid(row=12, column=1, sticky="w")
        self.effectiveness_slider = tk.Scale(self.main_frame, from_=0, to=100, resolution=1,
                                             orient="horizontal", command=self.update_vaccine_effectiveness, length=200)
        self.effectiveness_slider.set(self.vaccine_effectiveness * 100)
        self.effectiveness_slider.grid(row=13, column=1, sticky="ew")

        ttk.Label(self.main_frame, text="Mortality Rate").grid(row=14, column=1, sticky="w")
        self.mortality_slider = tk.Scale(self.main_frame, from_=0.0, to=0.1, resolution=0.001,
                                         orient="horizontal", command=self.update_mortality_rate, length=200)
        self.mortality_slider.set(self.mortality_rate)
        self.mortality_slider.grid(row=15, column=1, sticky="ew")

    # --- Updates from Sliders ---
    def update_infectivity(self, val):
        self.infectivity = float(val)

    def update_recovery_rate(self, val):
        self.recovery_rate = float(val)

    def update_waning_immunity_rate(self, val):
        self.waning_immunity_rate = float(val)

    def update_jump_infectivity(self, val):
        self.jump_infectivity = float(val)

    def update_agent_count(self, val):
        self.agent_count = int(val)
        self.reset_simulation()

    def update_vaccination_rate(self, val):
        self.vaccination_rate = int(val) / 100
        self.reset_simulation()

    def update_vaccine_effectiveness(self, val):
        self.vaccine_effectiveness = int(val) / 100
        self.reset_simulation()

    def update_mortality_rate(self, val):
        self.mortality_rate = float(val)

    # --- Simulation Logic ---
    def create_agents(self):
        """Initializes the agents' positions, status, and movement vectors."""
        self.agents = []
        for i in range(self.agent_count):
            status = 'S'  # S: Susceptible, I: Infected, R: Recovered
            is_vaccinated = False
            # Determine if agent starts vaccinated
            if random.random() < self.vaccination_rate:
                is_vaccinated = True

            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
            self.agents.append({'x': x, 'y': y, 'dx': dx, 'dy': dy, 'status': status, 'is_vaccinated': is_vaccinated})

        # Infect one agent to start the simulation
        if self.agents:
            # Pick a non-vaccinated agent to start the infection, if possible
            non_vaccinated = [a for a in self.agents if not a['is_vaccinated']]
            if non_vaccinated:
                random.choice(non_vaccinated)['status'] = 'I'
            else:
                self.agents[0]['status'] = 'I'

    def reset_simulation(self):
        """Resets the simulation to its initial state."""
        self.stop_simulation()
        self.create_agents()
        self.susceptible_count = []
        self.infected_count = []
        self.recovered_count = []
        self.vaccinated_count = []
        self.mortality_count = []
        self.total_removed = 0
        self.time_steps = []
        self.time = 0
        self.record_data()
        self.draw_plot()

    def step(self):
        """Performs one simulation step: movement and state changes."""
        # 1. Update positions
        for agent in self.agents:
            # Simple movement with toroidal boundary conditions
            agent['x'] = (agent['x'] + agent['dx'] + self.size) % self.size
            agent['y'] = (agent['y'] + agent['dy'] + self.size) % self.size
            # Randomly change direction occasionally
            if random.random() < 0.1:
                agent['dx'] = random.choice([-1, 0, 1])
                agent['dy'] = random.choice([-1, 0, 1])

        # 2. Handle state changes (Infection, Recovery, Waning Immunity, and Mortality)
        agents_to_remove = []

        # I -> M (Mortality)
        for agent in self.agents:
            if agent['status'] == 'I' and random.random() < self.mortality_rate:
                agents_to_remove.append(agent)

        # Remove dead agents from the simulation list and update total removed count
        self.total_removed += len(agents_to_remove)
        self.agents = [agent for agent in self.agents if agent not in agents_to_remove]

        # Check for interactions and apply other state changes
        # Use a list of changes to apply after the loop to avoid side effects
        changes = []
        infected_agents = [a for a in self.agents if a['status'] == 'I']
        susceptible_agents = [a for a in self.agents if a['status'] == 'S']
        recovered_agents = [a for a in self.agents if a['status'] == 'R']

        # S -> I: Contagion by proximity
        for i in range(len(infected_agents)):
            for j in range(len(susceptible_agents)):
                infected = infected_agents[i]
                susceptible = susceptible_agents[j]
                if infected['x'] == susceptible['x'] and infected['y'] == susceptible['y']:
                    infect_chance = self.infectivity
                    if susceptible['is_vaccinated']:
                        infect_chance *= (1 - self.vaccine_effectiveness)
                    if random.random() < infect_chance:
                        changes.append({'agent': susceptible, 'new_status': 'I'})

        # S -> I: Jump infection
        if infected_agents and susceptible_agents and random.random() < self.jump_infectivity:
            random_susceptible = random.choice(susceptible_agents)
            infect_chance = self.jump_infectivity
            if random_susceptible['is_vaccinated']:
                infect_chance *= (1 - self.vaccine_effectiveness)
            if random.random() < infect_chance:
                changes.append({'agent': random_susceptible, 'new_status': 'I'})

        # I -> R: Recovery
        for agent in infected_agents:
            if random.random() < self.recovery_rate:
                changes.append({'agent': agent, 'new_status': 'R'})

        # R -> S: Waning immunity
        for agent in recovered_agents:
            if random.random() < self.waning_immunity_rate:
                changes.append({'agent': agent, 'new_status': 'S'})

        # Apply changes
        for change in changes:
            change['agent']['status'] = change['new_status']

    def record_data(self):
        """Records the current counts of S, I, and R agents."""
        self.time_steps.append(self.time)
        self.susceptible_count.append(sum(1 for a in self.agents if a['status'] == 'S' and not a['is_vaccinated']))
        self.infected_count.append(sum(1 for a in self.agents if a['status'] == 'I'))
        self.recovered_count.append(sum(1 for a in self.agents if a['status'] == 'R'))
        self.vaccinated_count.append(sum(1 for a in self.agents if a['is_vaccinated']))
        self.mortality_count.append(self.total_removed)

    def draw_plot(self):
        """Draws the agent grid and the population curve plots."""
        # Clear plots
        self.ax_grid.clear()
        self.ax_curve.clear()

        # Grid Plot (Agent positions)
        s_agents = [(a['x'], a['y']) for a in self.agents if a['status'] == 'S' and not a['is_vaccinated']]
        v_agents = [(a['x'], a['y']) for a in self.agents if a['is_vaccinated'] and a['status'] == 'S']
        i_agents = [(a['x'], a['y']) for a in self.agents if a['status'] == 'I']
        r_agents = [(a['x'], a['y']) for a in self.agents if a['status'] == 'R']

        if s_agents:
            self.ax_grid.scatter([a[0] for a in s_agents], [a[1] for a in s_agents], c='green', s=10,
                                 label='Susceptible')
        if v_agents:
            self.ax_grid.scatter([a[0] for a in v_agents], [a[1] for a in v_agents], c='yellow', s=10,
                                 label='Vaccinated')
        if i_agents:
            self.ax_grid.scatter([a[0] for a in i_agents], [a[1] for a in i_agents], c='red', s=10, label='Infected')
        if r_agents:
            self.ax_grid.scatter([a[0] for a in r_agents], [a[1] for a in r_agents], c='blue', s=10, label='Recovered')

        self.ax_grid.set_xlim(0, self.size)
        self.ax_grid.set_ylim(0, self.size)
        self.ax_grid.set_title(f"Agent Distribution (Time: {self.time})")
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        self.ax_grid.legend(loc='upper right')

        # Logistic Curve Plot
        self.ax_curve.plot(self.time_steps, self.susceptible_count, label='Susceptible', color='green')
        self.ax_curve.plot(self.time_steps, self.vaccinated_count, label='Vaccinated', color='yellow')
        self.ax_curve.plot(self.time_steps, self.infected_count, label='Infected', color='red')
        self.ax_curve.plot(self.time_steps, self.recovered_count, label='Recovered', color='blue')
        self.ax_curve.plot(self.time_steps, self.mortality_count, label='Mortality', color='black')
        self.ax_curve.set_title("SIRS Population Curve")
        self.ax_curve.set_xlabel("Time Step")
        self.ax_curve.set_ylabel("Agent Count")
        self.ax_curve.legend()
        self.ax_curve.grid(True)

        self.canvas_grid.draw_idle()
        self.canvas_curve.draw_idle()

    # --- Play/Pause Functionality ---
    def toggle_simulation(self):
        if self.running:
            self.stop_simulation()
        else:
            if sum(1 for a in self.agents if a['status'] == 'I') == 0:
                self.reset_simulation()
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
        """The main simulation loop."""
        if self.running and self.time < self.max_time and sum(1 for a in self.agents if a['status'] == 'I') > 0:
            self.step()
            self.time += 1
            self.record_data()
            self.draw_plot()
            self.job = self.master.after(self.step_delay, self.run_step)
        else:
            self.stop_simulation()


if __name__ == "__main__":
    root = tk.Tk()
    app = SIRS_Movement_Simulation(root)
    root.mainloop()
