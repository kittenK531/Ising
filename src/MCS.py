import random

import numpy as np

from functions import (
    get_seed,
    initialize,
    make_GIF_local,
    preparation,
    save_frame,
    sequence_loop,
    visualize,
)


def tabulated_energy(beta, J=1.0):

    energy_record = np.zeros((5, 2))

    for idx, dEo in enumerate([-8, -4, 0, 4, 8]):
        energy_record[idx, :] = dEo, np.exp(-1 * beta * dEo * J)

    return energy_record


def get_energy(x, y, lattice, N):

    dxy = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])

    spin0 = lattice[int(x), int(y)]

    E = 0

    for j in dxy:

        cy, cx = y + j[0], x + j[1]

        spinc = lattice[int(cy), int(cx)]

        if spinc < 1:
            spinc = -1

        E -= spin0 * spinc

    return E


def get_Boltzmann_factor(E_diff, energy2table):

    for idx, E_ref in enumerate(energy2table[:, 0]):
        if E_diff == E_ref:
            return energy2table[idx, 1]


def MCS(N, beta, J, lattice, seed_y, seed_x, cluster_list, flipped):

    x, y = seed_y, seed_x
    Eo = get_energy(x, y, lattice, N)
    lattice[int(x), int(y)] = lattice[int(x), int(y)] * -1  # flip
    Ef = get_energy(x, y, lattice, N)
    lattice[int(x), int(y)] = lattice[int(x), int(y)] * -1

    """ Step 2: calculate the energy fluctuation """
    energy2table = tabulated_energy(beta, J)
    E_diff = Ef - Eo

    if (E_diff < 0) or random.uniform(0, 1) < get_Boltzmann_factor(
        E_diff, energy2table
    ):
        lattice[int(x), int(y)] = lattice[int(x), int(y)] * -1
        new_cluster_mem = np.array([[int(y), int(x)]])
        cluster_list = np.vstack((cluster_list, new_cluster_mem))
        flipped[int(y), int(x)] = 1

    """
    print(f"Energy difference is {E_diff}")
    print(f"The boltzmann factor is {get_Boltzmann_factor(E_diff, energy2table)}.")
    """

    return lattice, cluster_list, flipped


def iterative(N, beta, J, lattice, seed_y, seed_x, cluster_list, flipped):

    x, y = seed_y, seed_x

    array = sequence_loop(N, x, y)

    for i in array:

        x, y = int(i[0]), int(i[1])

        lattice, cluster_list, flipped = MCS(
            N, beta, J, lattice, x, y, cluster_list, flipped
        )

    return lattice, flipped


def run(N, beta, J, iterations):

    seed_y, seed_x = get_seed(N)

    lattice, flipped = initialize(N, seed_x, seed_y)

    visualize(N, lattice, f"init", folder="record_local")

    save_frame(N, lattice, "flipped", iteration=0, folder="record_local")

    for i in range(iterations):

        print(f"Overall MCS teration: {i+1}")

        cluster_list, crystal = preparation(lattice, seed_x, seed_y)

        lattice, flipped = iterative(
            N, beta, J, lattice, seed_y, seed_x, cluster_list, flipped
        )

        visualize(N, lattice, "flipped", folder="record_local")

        save_frame(N, lattice, "flipped", iteration=i + 1, folder="record_local")

        seed_y, seed_x = get_seed(N)

    make_GIF_local(N, beta, J, "flipped", clean=True)


""" Execution """
J = 1.0
N = 50
beta = 0.2

# Note: b = 0.2 converge on 3rd run
# TODO: animate and while loop

run(N, beta, J, 10)
