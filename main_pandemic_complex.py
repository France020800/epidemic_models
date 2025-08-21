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


class Pandemic_Simulation_Lockdown:
    """
    A GUI application to run and visualize a pandemic simulation with a more
    realistic social structure, including homes, workplaces, and a lockdown
    mechanism.
    """

    def __init__(self, master):
        self.master = master
        master.title("Pandemic Simulation with Lockdown")

        # --- Simulation Parameters ---
        self.size = 100
        self.agent_count = 1000
        self.infectivity = 0.3
        self.recovery_rate = 0.05
        self.waning_immunity_rate = 0.005
        self.jump_infectivity = 0.0005
        self.vaccination_rate = 0.5
        self.vaccine_effectiveness = 0.8
        self.mortality_rate = 0.005
        self.workplace_count = 15
        self.lockdown_threshold = 0.1  # % of population infected to trigger lockdown
        self.lockdown_duration = 50  # number of steps lockdown lasts
        self.time = 0
        self.max_time = 1000

        # --- Simulation State ---
        self.running = False
        self.job = None
        self.step_delay = 50
        self.agents = []
        self.workplaces = []
        self.lockdown_active = False
        self.lockdown_timer = 0

        # --- Data for the Logistic Curve ---
        self.susceptible_count = []
        self.infected_count = []
        self.recovered_count = []
        self.vaccinated_count = []
        self.mortality_count = []
        self.time_steps = []
        self.total_removed = 0

        # --- GUI Elements ---
        self.main_frame = ttk.Frame(master, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- Matplotlib Figures ---
        self.fig_grid, self.ax_grid = plt.subplots(figsize=(5, 5))
        self.canvas_grid = FigureCanvasTkAgg(self.fig_grid, self.main_frame)
        self.canvas_grid.get_tk_widget().grid(row=0, column=0, rowspan=18, padx=10, pady=10)

        self.fig_curve, self.ax_curve = plt.subplots(figsize=(5, 3))
        self.canvas_curve = FigureCanvasTkAgg(self.fig_curve, self.main_frame)
        self.canvas_curve.get_tk_widget().grid(row=18, column=0, rowspan=2, padx=10, pady=10)

        # --- Sliders and Labels for Parameters ---
        self.create_sliders()

        # --- Control Buttons and Status Label ---
        self.control_button = ttk.Button(self.main_frame, text="Play", command=self.toggle_simulation)
        self.control_button.grid(row=18, column=1, sticky="ew", pady=5)

        self.reset_button = ttk.Button(self.main_frame, text="Reset Simulation", command=self.reset_simulation)
        self.reset_button.grid(row=19, column=1, sticky="ew", pady=5)

        self.status_label = ttk.Label(self.main_frame, text="Status: Ready")
        self.status_label.grid(row=20, column=1, sticky="w", pady=5)

        # Initialize simulation
        self.reset_simulation()
        self.draw_plot()

    def create_sliders(self):
        """Creates the sliders and labels for all simulation parameters."""
        row_num = 0
        ttk.Label(self.main_frame, text="Agent Count").grid(row=row_num, column=1, sticky="w")
        self.agent_count_slider = tk.Scale(self.main_frame, from_=100, to=5000, resolution=100,
                                           orient="horizontal", command=self.update_agent_count, length=200)
        self.agent_count_slider.set(self.agent_count)
        self.agent_count_slider.grid(row=row_num + 1, column=1, sticky="ew")

        row_num += 2
        ttk.Label(self.main_frame, text="Infectivity").grid(row=row_num, column=1, sticky="w")
        self.infectivity_slider = tk.Scale(self.main_frame, from_=0.0, to=1.0, resolution=0.01,
                                           orient="horizontal", command=self.update_infectivity, length=200)
        self.infectivity_slider.set(self.infectivity)
        self.infectivity_slider.grid(row=row_num + 1, column=1, sticky="ew")

        row_num += 2
        ttk.Label(self.main_frame, text="Recovery Rate").grid(row=row_num, column=1, sticky="w")
        self.recovery_slider = tk.Scale(self.main_frame, from_=0.0, to=1.0, resolution=0.01,
                                        orient="horizontal", command=self.update_recovery_rate, length=200)
        self.recovery_slider.set(self.recovery_rate)
        self.recovery_slider.grid(row=row_num + 1, column=1, sticky="ew")

        row_num += 2
        ttk.Label(self.main_frame, text="Waning Immunity").grid(row=row_num, column=1, sticky="w")
        self.waning_immunity_slider = tk.Scale(self.main_frame, from_=0.0, to=0.2, resolution=0.001,
                                               orient="horizontal", command=self.update_waning_immunity_rate,
                                               length=200)
        self.waning_immunity_slider.set(self.waning_immunity_rate)
        self.waning_immunity_slider.grid(row=row_num + 1, column=1, sticky="ew")

        row_num += 2
        ttk.Label(self.main_frame, text="Mortality Rate").grid(row=row_num, column=1, sticky="w")
        self.mortality_slider = tk.Scale(self.main_frame, from_=0.0, to=0.1, resolution=0.001,
                                         orient="horizontal", command=self.update_mortality_rate, length=200)
        self.mortality_slider.set(self.mortality_rate)
        self.mortality_slider.grid(row=row_num + 1, column=1, sticky="ew")

        row_num += 2
        ttk.Label(self.main_frame, text="Vaccination Rate (%)").grid(row=row_num, column=1, sticky="w")
        self.vaccination_slider = tk.Scale(self.main_frame, from_=0, to=100, resolution=1,
                                           orient="horizontal", command=self.update_vaccination_rate, length=200)
        self.vaccination_slider.set(self.vaccination_rate * 100)
        self.vaccination_slider.grid(row=row_num + 1, column=1, sticky="ew")

        row_num += 2
        ttk.Label(self.main_frame, text="Vaccine Effectiveness (%)").grid(row=row_num, column=1, sticky="w")
        self.effectiveness_slider = tk.Scale(self.main_frame, from_=0, to=100, resolution=1,
                                             orient="horizontal", command=self.update_vaccine_effectiveness, length=200)
        self.effectiveness_slider.set(self.vaccine_effectiveness * 100)
        self.effectiveness_slider.grid(row=row_num + 1, column=1, sticky="ew")

        row_num += 2
        ttk.Label(self.main_frame, text="Lockdown Threshold (%)").grid(row=row_num, column=1, sticky="w")
        self.lockdown_threshold_slider = tk.Scale(self.main_frame, from_=0, to=100, resolution=1,
                                                  orient="horizontal", command=self.update_lockdown_threshold,
                                                  length=200)
        self.lockdown_threshold_slider.set(self.lockdown_threshold * 100)
        self.lockdown_threshold_slider.grid(row=row_num + 1, column=1, sticky="ew")

        row_num += 2
        ttk.Label(self.main_frame, text="Lockdown Duration (steps)").grid(row=row_num, column=1, sticky="w")
        self.lockdown_duration_slider = tk.Scale(self.main_frame, from_=10, to=200, resolution=10,
                                                 orient="horizontal", command=self.update_lockdown_duration, length=200)
        self.lockdown_duration_slider.set(self.lockdown_duration)
        self.lockdown_duration_slider.grid(row=row_num + 1, column=1, sticky="ew")

    # --- Updates from Sliders ---
    def update_agent_count(self, val):
        self.agent_count = int(val)
        self.reset_simulation()

    def update_infectivity(self, val):
        self.infectivity = float(val)

    def update_recovery_rate(self, val):
        self.recovery_rate = float(val)

    def update_waning_immunity_rate(self, val):
        self.waning_immunity_rate = float(val)

    def update_mortality_rate(self, val):
        self.mortality_rate = float(val)

    def update_vaccination_rate(self, val):
        self.vaccination_rate = int(val) / 100
        self.reset_simulation()

    def update_vaccine_effectiveness(self, val):
        self.vaccine_effectiveness = int(val) / 100

    def update_lockdown_threshold(self, val):
        self.lockdown_threshold = int(val) / 100

    def update_lockdown_duration(self, val):
        self.lockdown_duration = int(val)

    # --- Simulation Logic ---
    def create_agents(self):
        """Initializes the agents' positions, status, and destinations."""
        self.agents = []
        self.workplaces = []

        # Create workplaces
        for i in range(self.workplace_count):
            wx = random.randint(0, self.size - 1)
            wy = random.randint(0, self.size - 1)
            self.workplaces.append({'x': wx, 'y': wy})

        # Create agents with homes and assigned workplaces
        for i in range(self.agent_count):
            status = 'S'
            is_vaccinated = False
            if random.random() < self.vaccination_rate:
                is_vaccinated = True

            # Assign a random home
            home_x = random.randint(0, self.size - 1)
            home_y = random.randint(0, self.size - 1)

            # Assign a random workplace from the list
            workplace = random.choice(self.workplaces)
            work_x = workplace['x']
            work_y = workplace['y']

            self.agents.append({
                'status': status,
                'is_vaccinated': is_vaccinated,
                'home_x': home_x,
                'home_y': home_y,
                'work_x': work_x,
                'work_y': work_y,
                'current_x': home_x,
                'current_y': home_y
            })

        # Infect one agent to start the simulation
        if self.agents:
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
        self.time_steps = []
        self.total_removed = 0
        self.time = 0
        self.lockdown_active = False
        self.lockdown_timer = 0
        self.status_label.config(text="Status: Ready")
        self.record_data()
        self.draw_plot()

    def step(self):
        """Performs one simulation step: movement and state changes."""

        # --- Check for and handle lockdown state ---
        infected_pop = sum(1 for a in self.agents if a['status'] == 'I')
        if infected_pop / self.agent_count > self.lockdown_threshold and not self.lockdown_active:
            self.lockdown_active = True
            self.lockdown_timer = self.lockdown_duration
            self.status_label.config(text="Status: Lockdown Active")

        if self.lockdown_active:
            self.lockdown_timer -= 1
            if self.lockdown_timer <= 0:
                self.lockdown_active = False
                self.status_label.config(text="Status: Lockdown Ended")

        # --- Update agent positions based on daily cycle and lockdown status ---
        for agent in self.agents:
            if self.lockdown_active:
                # During lockdown, agents stay at home and jitter slightly
                agent['current_x'] = agent['home_x'] + random.choice([-1, 0, 1])
                agent['current_y'] = agent['home_y'] + random.choice([-1, 0, 1])

            else:
                # Normal movement cycle (commute)
                # Morning commute (e.g., first 10 steps of a 20-step cycle)
                if (self.time % 20) < 10:
                    # Move towards workplace
                    if agent['current_x'] != agent['work_x']:
                        agent['current_x'] += 1 if agent['work_x'] > agent['current_x'] else -1
                    if agent['current_y'] != agent['work_y']:
                        agent['current_y'] += 1 if agent['work_y'] > agent['current_y'] else -1
                else:
                    # Evening commute (e.g., last 10 steps of a 20-step cycle)
                    # Move towards home
                    if agent['current_x'] != agent['home_x']:
                        agent['current_x'] += 1 if agent['home_x'] > agent['current_x'] else -1
                    if agent['current_y'] != agent['home_y']:
                        agent['current_y'] += 1 if agent['home_y'] > agent['current_y'] else -1

            # Ensure agents are within grid bounds
            agent['current_x'] = (agent['current_x'] + self.size) % self.size
            agent['current_y'] = (agent['current_y'] + self.size) % self.size

        # --- Handle state changes (Infection, Recovery, Waning Immunity, and Mortality) ---
        agents_to_remove = []

        # I -> M (Mortality)
        for agent in self.agents:
            if agent['status'] == 'I' and random.random() < self.mortality_rate:
                agents_to_remove.append(agent)

        # Remove dead agents and update total removed count
        self.total_removed += len(agents_to_remove)
        self.agents = [agent for agent in self.agents if agent not in agents_to_remove]

        # Use a list of changes to apply after the loops to avoid side effects
        changes = []
        infected_agents = [a for a in self.agents if a['status'] == 'I']
        susceptible_agents = [a for a in self.agents if a['status'] == 'S']
        recovered_agents = [a for a in self.agents if a['status'] == 'R']

        # S -> I: Contagion by proximity
        for i in range(len(infected_agents)):
            for j in range(len(susceptible_agents)):
                infected = infected_agents[i]
                susceptible = susceptible_agents[j]

                # Check if agents are at the same location
                if infected['current_x'] == susceptible['current_x'] and \
                        infected['current_y'] == susceptible['current_y']:
                    infect_chance = self.infectivity
                    if susceptible['is_vaccinated']:
                        infect_chance *= (1 - self.vaccine_effectiveness)
                    if random.random() < infect_chance:
                        changes.append({'agent': susceptible, 'new_status': 'I'})

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
        """Records the current counts of S, I, R, V, and M agents."""
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
        s_agents = [(a['current_x'], a['current_y']) for a in self.agents if
                    a['status'] == 'S' and not a['is_vaccinated']]
        v_agents = [(a['current_x'], a['current_y']) for a in self.agents if a['is_vaccinated'] and a['status'] == 'S']
        i_agents = [(a['current_x'], a['current_y']) for a in self.agents if a['status'] == 'I']
        r_agents = [(a['current_x'], a['current_y']) for a in self.agents if a['status'] == 'R']

        # Draw homes and workplaces
        home_locations = [(a['home_x'], a['home_y']) for a in self.agents]
        workplace_locations = [(w['x'], w['y']) for w in self.workplaces]

        self.ax_grid.scatter([h[0] for h in home_locations], [h[1] for h in home_locations],
                             c='grey', s=15, alpha=0.3, label='Homes')
        self.ax_grid.scatter([w[0] for w in workplace_locations], [w[1] for w in workplace_locations],
                             c='purple', marker='s', s=40, label='Workplaces')

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
        title = f"Agent Distribution (Time: {self.time})"
        if self.lockdown_active:
            title += f" - LOCKDOWN ({self.lockdown_timer} steps left)"
        self.ax_grid.set_title(title)
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
            self.status_label.config(text="Status: Running")
            self.run_step()

    def stop_simulation(self):
        if self.running:
            self.running = False
            self.control_button.config(text="Play")
            self.status_label.config(text="Status: Paused")
            if self.job is not None:
                self.master.after_cancel(self.job)
                self.job = None

    def run_step(self):
        """The main simulation loop."""
        if self.running and self.time < self.max_time:
            self.step()
            self.time += 1
            self.record_data()
            self.draw_plot()
            self.job = self.master.after(self.step_delay, self.run_step)
        else:
            self.stop_simulation()


if __name__ == "__main__":
    root = tk.Tk()
    app = Pandemic_Simulation_Lockdown(root)
    root.mainloop()