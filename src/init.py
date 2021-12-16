import random

import numpy as np

from functions import (
    animate,
    flip,
    get_index_outer,
    get_P_add,
    get_seed,
    get_seed2b_list,
    initialize,
    is_neighbour,
    make_GIF,
    not_terminate,
    preparation,
    sequence_loop,
    visualize,
)


def growth(sequence, lattice, spin0, Pr, cluster_list, crystal):

    for c in sequence:

        cy, cx = int(c[0]), int(c[1])

        spin_c = lattice[cy, cx]

        if is_neighbour(cy, cx, crystal):
            """
            print(
                f"Current coord {cy, cx} is a neigbour {is_neighbour(cy, cx, crystal)}"
            )
            """

            if spin0 * spin_c > 0 and random.uniform(0, 1) < Pr:  # TODO: Check physics

                new_cluster_mem = np.array([[cy, cx]])
                cluster_list = np.vstack((cluster_list, new_cluster_mem))
                crystal[cy, cx] = 1

    return cluster_list


def iterative(
    N,
    start_x,
    start_y,
    Pr,
    cluster_list,
    flipped,
    lattice_pos,
    crystal,
    previous_count,
):

    seed_y, seed_x = start_y, start_x
    lattice = lattice_pos

    spin0 = lattice[seed_y, seed_x]

    animate(N, lattice, flipped, previous_count, 0)

    i = 0

    while not_terminate(lattice, cluster_list):
        array = sequence_loop(N, seed_y, seed_x)
        cluster_list = growth(array, lattice, spin0, Pr, cluster_list, crystal)
        # print(f"Cluster: {cluster_list}")
        lattice, flipped = flip(lattice, cluster_list, seed_x, seed_y, flipped)
        visualize(N, lattice, "flipped")
        visualize(N, flipped, "crystal")

        animate(N, lattice, flipped, previous_count, iterations=i + 1)

        index_arr = get_index_outer(cluster_list)
        # print(index_arr)
        r = random.randint(0, len(index_arr) - 1)
        seed_y, seed_x = cluster_list[r, :]
        print(f"iter: {i+1} Seed: {seed_y, seed_x}")
        i += 1

    iterations = i

    return lattice, flipped, iterations


def whole_growth(N, beta, J):

    seed_y, seed_x = get_seed(N)

    lattice, flipped = initialize(N, seed_x, seed_y)

    previous_count = 0

    unflipp_num = 1

    i = 0

    while unflipp_num > 0:

        print(f"Overall cluster iteration: {i+1}")

        cluster_list, crystal = preparation(lattice, seed_x, seed_y)

        if i == 0:
            visualize(N, lattice, f"init")

        Pr = get_P_add(beta, J)

        lattice, flipped, cluster_iter = iterative(
            N,
            seed_x,
            seed_y,
            Pr,
            cluster_list,
            flipped,
            lattice,
            crystal,
            previous_count,
        )

        unflipped = get_seed2b_list(N, lattice)

        previous_count += cluster_iter

        if len(unflipped) - 1 >= 0:
            r = random.randint(0, len(unflipped) - 1)

            seed_y, seed_x = int(unflipped[r, 0]), int(unflipped[r, 1])

        else:
            break

        i += 1

    previous_count += i

    make_GIF(N, previous_count, clean=True)


""" Execution """
J = 1.0
N = 5
beta = 0.2

whole_growth(N, beta, J)
