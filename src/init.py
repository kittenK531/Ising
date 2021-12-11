import random

import numpy as np

from functions import (
    flip,
    get_index_outer,
    get_neighbour_list,
    get_outer_list,
    get_P_add,
    is_neighbour,
    sequence_loop,
    visualize,
)


def preparation(N):

    seed_x, seed_y = random.randint(1, N - 2), random.randint(1, N - 2)

    # lattice = initialize(N)
    lattice = np.ones((N + 2, N + 2))
    lattice[N, N] = -1

    spin0 = lattice[seed_y, seed_x]

    lattice = lattice * spin0  # for visualisation (1: spin0)

    """ Global array """

    cluster_list = np.ones((1, 2))
    cluster_list[0, :] = seed_y, seed_x

    flipped = np.ones(lattice.shape) * -1

    crystal = np.zeros(lattice.shape)  # for neighbour
    crystal[seed_y, seed_x] = 1

    return lattice, seed_y, seed_x, cluster_list, flipped, crystal


def growth(sequence, lattice, spin0, Pr, cluster_list, crystal):

    for c in sequence:

        cy, cx = int(c[0]), int(c[1])

        outer_list = get_outer_list(cluster_list)
        # print(f"outer = {outer_list}")

        neighbour = get_neighbour_list(cy, cx)
        spin_c = lattice[cy, cx]

        if is_neighbour(cy, cx, crystal):
            print(
                f"Current coord {cy, cx} is a neigbour {is_neighbour(cy, cx, crystal)}"
            )

            if spin0 * spin_c > 0 and random.uniform(0, 1) < Pr:  # TODO: Check physics

                new_cluster_mem = np.array([[cy, cx]])
                cluster_list = np.vstack((cluster_list, new_cluster_mem))
                crystal[cy, cx] = 1

    return cluster_list


def iterative(N, start_x, start_y, Pr, cluster_list, flipped, lattice_pos, crystal):

    seed_y, seed_x = start_y, start_x
    lattice = lattice_pos

    spin0 = lattice[seed_y, seed_x]

    for i in range(20):
        array = sequence_loop(N, seed_y, seed_x)
        cluster_list = growth(array, lattice, spin0, Pr, cluster_list, crystal)
        print(f"Cluster: {cluster_list}")
        lattice, flipped = flip(lattice, cluster_list, seed_x, seed_y, flipped)
        visualize(N, lattice, f"flipped_{N}")
        visualize(N, flipped, f"crystal_{N}")

        index_arr = get_index_outer(cluster_list)
        print(index_arr)
        r = random.randint(0, len(index_arr) - 1)
        seed_y, seed_x = cluster_list[r, :]
        print(f"Seed: {seed_y, seed_x}")


""" Execution """
J = 1.0
N = 10
beta = 0.2

# seed_x, seed_y = debug_mode(J, N, beta)
lattice, seed_y, seed_x, cluster_list, flipped, crystal = preparation(N)

visualize(N, lattice, f"init_{N}")

Pr = get_P_add(beta, J)

iterative(N, seed_x, seed_y, Pr, cluster_list, flipped, lattice, crystal)
