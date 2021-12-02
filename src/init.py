import random

import numpy as np


def initialize(N):
    """Produces a 2d array containing position of square array
    N: number of particles per axis

    return [N * N] array size numpy array
    """

    lattice = np.zeros((N + 2, N + 2))

    for i in range(1, N + 1):
        for j in range(1, N + 1):

            r = random.randint(0, 1)
            lattice[i, j] = r if (r > 0.5) else -1

    print(f"real lattice:\n {lattice}")

    """ Periodic Boundary condition """
    lattice[-1, :] = lattice[1, :]
    lattice[:, -1] = lattice[:, 1]
    lattice[0, :] = lattice[N - 1, :]
    lattice[:, 0] = lattice[:, N - 1]
    print(f"After boundary condition, the lattice becomes:\n {lattice}")

    return lattice


def tabulated_energy(beta):

    energy_record = np.zeros((5, 2))

    for idx, dEo in enumerate([-8, -4, 0, 4, 8]):
        energy_record[idx, :] = dEo, np.exp(-1 * beta * dEo)

    return energy_record


def get_energy(x, y, lattice, N):

    top = lattice[int(x), int(y) + 1]
    bottom = lattice[int(x), int(y) - 1]
    left = lattice[int(x) - 1, int(y)]
    right = lattice[int(x) + 1, int(y)]

    interaction_E = -1 * lattice[int(x), int(y)] * (top + bottom + left + right)

    return interaction_E


def get_Boltzmann_factor(E_diff, energy2table):

    for idx, E_ref in enumerate(energy2table[:, 0]):
        if E_diff == E_ref:
            return energy2table[idx, 1]


def operation(N, beta):

    """Step 0: initialize cubic lattice with random spin configuration"""
    lattice = initialize(N)

    """ Step 1: choose a random lattice point spin to flip as cluster point """
    flipped = np.zeros((N ** 2, 2))  # max number flipped is N ** 2
    flipped[0, :] = random.randint(1, N - 2), random.randint(
        1, N - 2
    )  # record the coord of the flipped spin

    x, y = flipped[0, :]
    Eo = get_energy(x, y, lattice, N)

    lattice[int(x), int(y)] = lattice[int(x), int(y)] * -1  # flip
    Ef = get_energy(x, y, lattice, N)

    """ Step 2: calculate the energy fluctuation """
    energy2table = tabulated_energy(beta)
    E_diff = Ef - Eo

    print(get_Boltzmann_factor(E_diff, energy2table))


""" Execution """
operation(5, 0.5)
