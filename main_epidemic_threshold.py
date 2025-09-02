import numpy as np
import matplotlib.pyplot as plt


# The SIR simulation function
def run_sir_simulation(
        total_population,
        population_density,
        initial_infected=1,
        infectivity=0.3,
        recovery_rate=0.1,
        waning_immunity_rate=0.01
):
    """
    Simulates a SIR epidemic and returns the total percentage of the population
    that was ever infected.

    Args:
        total_population (int): The total number of people in the simulation.
        population_density (float): The population density (0.0 to 1.0).
        initial_infected (int): The starting number of infected individuals.
        infectivity (float): The rate at which an infected person can infect a susceptible person.
        recovery_rate (float): The rate at which an infected person recovers.
        waning_immunity_rate (float): The rate at which a recovered person becomes susceptible again.

    Returns:
        float: The final percentage of the population that was ever infected.
    """
    # Initialize the population states
    S = total_population - initial_infected
    I = initial_infected
    R = 0

    # Track the cumulative number of people who have ever been infected
    cumulative_infected = initial_infected

    # Run the simulation until no infected individuals remain
    while I > 0:
        # Calculate new infections based on the current state and density
        new_infections = infectivity * I * (S / total_population) * population_density

        # Calculate new recoveries
        new_recoveries = recovery_rate * I

        # Calculate people losing immunity
        new_waning = waning_immunity_rate * R

        # Update the population states, ensuring they don't go below zero
        actual_new_infections = min(new_infections, S)

        # Update S
        S = S - actual_new_infections + new_waning

        # Update R
        R = R - new_waning + new_recoveries

        # Update I
        I = I + actual_new_infections - new_recoveries

        # A small check to prevent floating point issues from causing an infinite loop
        if I < 0.5:
            I = 0

        cumulative_infected += actual_new_infections

        # Ensure the total population remains constant
        total = S + I + R
        if total != total_population:
            # Distribute any small floating point errors
            S += total_population - total

    return (cumulative_infected / total_population) * 100


def main():
    """
    Main function to run the simulations and plot the results.
    """
    # Fixed parameters for the simulation
    TOTAL_POPULATION = 100000
    INFECTIVITY = 0.5
    RECOVERY_RATE = 0.2
    WANING_IMMUNITY_RATE = 0.0

    # List of population densities to test, from 0.1 to 1.0
    population_densities = np.arange(0.1, 1.1, 0.1)

    # Store the results of each simulation
    infected_percentages = []

    # Run a simulation for each population density
    for density in population_densities:
        print(f"Running simulation for density: {density:.1f}")
        infected_percent = run_sir_simulation(
            total_population=TOTAL_POPULATION,
            population_density=density,
            infectivity=INFECTIVITY,
            recovery_rate=RECOVERY_RATE,
            waning_immunity_rate=WANING_IMMUNITY_RATE,
        )
        infected_percentages.append(infected_percent)

    # Plot the results
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(10, 6))
    plt.plot(population_densities, infected_percentages, marker='o', linestyle='-', color='#1f77b4')
    plt.title('Impact of Population Density on Total Infections')
    plt.xlabel('Population Density')
    plt.ylabel('Total Percentage of Population Ever Infected (%)')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
