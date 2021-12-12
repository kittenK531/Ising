import random

import numpy as np

from functions import (
    combine,
    flip,
    get_index_outer,
    get_P_add,
    is_neighbour,
    make_GIF,
    preparation,
    save_frame,
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
    N, start_x, start_y, Pr, cluster_list, flipped, lattice_pos, crystal, iterations=10
):

    seed_y, seed_x = start_y, start_x
    lattice = lattice_pos

    spin0 = lattice[seed_y, seed_x]

    save_frame(N, lattice, "flipped", iteration=0)
    save_frame(N, flipped, "crystal", iteration=0)
    combine(N, iteration=0)

    for i in range(iterations):
        array = sequence_loop(N, seed_y, seed_x)
        cluster_list = growth(array, lattice, spin0, Pr, cluster_list, crystal)
        # print(f"Cluster: {cluster_list}")
        lattice, flipped = flip(lattice, cluster_list, seed_x, seed_y, flipped)
        visualize(N, lattice, "flipped")
        visualize(N, flipped, "crystal")

        """ animate """
        save_frame(N, lattice, "flipped", iteration=i + 1)
        save_frame(N, flipped, "crystal", iteration=i + 1)

        combine(N, iteration=i + 1)

        index_arr = get_index_outer(cluster_list)
        # print(index_arr)
        r = random.randint(0, len(index_arr) - 1)
        seed_y, seed_x = cluster_list[r, :]
        print(f"Seed: {seed_y, seed_x}")

    make_GIF(N, iterations)


""" Execution """
J = 1.0
N = 50
beta = 0.2

lattice, seed_y, seed_x, cluster_list, flipped, crystal = preparation(N)

visualize(N, lattice, f"init")

Pr = get_P_add(beta, J)

iterative(N, seed_x, seed_y, Pr, cluster_list, flipped, lattice, crystal, iterations=20)
