import numpy as np
import matplotlib.pyplot as plt


def domany_kinzel_steady_state(p1, p2):
    """
    Calculates the steady-state fraction of infected sites
    for the Domany-Kinzel model using the mean-field equation.

    Args:
        p1 (float): Probability of infection with one infected neighbor.
        p2 (float): Probability of infection with two infected neighbors.

    Returns:
        float: The steady-state fraction of infected sites. Returns 0 if
               the epidemic dies out.
    """
    if (2 * p1 - p2) == 0:
        return 0

    # Calculate the non-trivial solution for x
    x = (2 * p1 - 1) / (2 * p1 - p2)

    # The steady-state fraction must be between 0 and 1.
    # The epidemic dies out if x is not positive.
    if x < 0 or x > 1:
        return 0

    # Check for the trivial solution x=0 (no infection)
    # The epidemic will only occur if the non-trivial solution is stable
    # which is true for p1 >= 0.5 according to the document
    if p1 < 0.5:
        return 0

    return x

if __name__ == "__main__":
    # Define the ranges for P1 and P2
    p1_values = np.linspace(0.0, 1.0, 100)
    p2_values = np.linspace(0.0, 1.0, 100)

    # Create a grid for P1 and P2
    P1, P2 = np.meshgrid(p1_values, p2_values)

    # Calculate the steady-state infection fraction for each combination of P1 and P2
    steady_state_fraction = np.zeros_like(P1)
    for i in range(P1.shape[0]):
        for j in range(P1.shape[1]):
            steady_state_fraction[i, j] = domany_kinzel_steady_state(P1[i, j], P2[i, j])

    # Plot the phase diagram
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(P1, P2, steady_state_fraction, cmap='viridis')
    plt.colorbar(label='Steady-State Infected Fraction (x)')
    plt.xlabel('Probability of infection with one neighbor (P1)')
    plt.ylabel('Probability of infection with two neighbors (P2)')
    plt.title('Domany-Kinzel Model Mean-Field Phase Diagram')
    plt.grid(True)
    plt.show()