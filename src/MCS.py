import random

import numpy as np

from functions import check_element, initialize, print_real_lattice


def tabulated_energy(beta, J=1.0):

    energy_record = np.zeros((5, 2))

    for idx, dEo in enumerate([-8, -4, 0, 4, 8]):
        energy_record[idx, :] = dEo, np.exp(-1 * beta * dEo * J)

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


def MCS(N, beta, J):

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
    energy2table = tabulated_energy(beta, J)
    E_diff = Ef - Eo

    print(f"Energy difference is {E_diff}")

    print(f"Tabulated energy array is {energy2table}")
    print(f"The boltzmann factor is {get_Boltzmann_factor(E_diff, energy2table)}.")


MCS(5, 0.5, 1)

""" back up """


def sequence_loop(N, start_x, start_y):

    # enter the start of x, y in x+1, y+1 in manual use

    array = np.zeros((N + 2, N + 2))

    iteration_count, current_count, visit_number = 1, 1, 0

    visited_list = np.ones((N * N, 2))
    visited_list[:, 0], visited_list[:, 1] = start_y, start_x

    x, y = start_x, start_y
    cx, cy = x, y

    dxy = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])

    for i in range(N * N - 1):

        for j in dxy:

            cx = x + j[1]
            cy = y + j[0]

            # print(y, x, cy, cx, not check_element(cx, cy, visited_list), (cx < N+1), (cy < N+1), current_count < (N*N), (cx > 0), (cy > 0))

            if (
                not check_element(cy, cx, visited_list)
                and (cx < N + 1)
                and (cy < N + 1)
                and current_count < ((N + 2) * (N + 2))
                and (cx > 0)
                and (cy > 0)
            ):
                """
                print(
                    f"Index: {cx, cy}, current count: {current_count}, iteration count: {iteration_count}, using xy from visited list [{visit_number}] of {y,x}"
                )
                """
                array[int(cy), int(cx)] = current_count  # order
                current_count += 1
                visited_list[current_count - 1, :] = int(cy), int(cx)
                # print(array)

            iteration_count += 1

        visit_number += 1
        y, x = visited_list[visit_number, :]

    # print(visited_list)

    print_real_lattice(N, array, islattice=False)

    return array
