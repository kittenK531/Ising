""" back up """
import random

import numpy as np
from tqdm import tqdm

from functions import (
    check_element,
    flip,
    get_index_outer,
    get_P_add,
    print_real_lattice,
    visualize,
)


def wolff_seq(N, spin0, seed_x, seed_y, beta, J, lattice, cluster_list):

    # enter the start of x, y in x+1, y+1 in manual use

    P_add = get_P_add(beta, J)
    array = np.zeros((N + 2, N + 2))  # Order

    iteration_count, current_count, visit_number, cluster_count, count = 1, 1, 0, 1, 1

    start_x, start_y = seed_x, seed_y

    visited_list = np.ones((N * N, 2))
    visited_list[:, 0], visited_list[:, 1] = start_y, start_x

    """
    cluster_list = np.ones((N * N, 2))
    cluster_list[:, 0], cluster_list[:, 1] = start_y, start_x
    cluster_list[0, :] = seed_y, seed_x
    """

    x, y = start_x, start_y
    cx, cy = x, y

    dxy = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])

    print(f"Seed: {seed_y, seed_x}")

    for i in tqdm(range(N * N - 1)):

        for j in dxy:

            cx = x + j[1]
            cy = y + j[0]

            if (
                not check_element(cy, cx, visited_list)
                and (cx < N + 1)
                and (cy < N + 1)
                and current_count < ((N + 2) * (N + 2))
                and (cx > 0)
                and (cy > 0)
                # and (not match(cy, cx, visited_list[0])) #TODO: check if needed
            ):

                """
                not_neigh_cluster = (
                    check_element(cy, cx, cluster_list) == None
                )  # [2] not operate if current neighbour is cluster member

                print(f"current ({cy, cx}), neighbour not cluster: {not_neigh_cluster}")
                """

                if check_element(y, x, cluster_list):
                    cspin = lattice[int(cy), int(cx)]

                    if (spin0 * cspin > 0) and (random.uniform(0, 1) < P_add):
                        cluster_list[cluster_count, :] = int(cy), int(cx)
                        cluster_count += 1

                    current_count += 1
                    visited_list[current_count - 1, :] = int(cy), int(cx)

                array[int(cy), int(cx)] = count  # order
                print(f"coord {cy, cx} count: {count}")
                count += 1
            iteration_count += 1

        visit_number += 1
        y, x = visited_list[visit_number, :]  # order

    print_real_lattice(N, array, printf=False, Word="After this iteration, lattice")

    return cluster_list, cluster_count  # [:cluster_count]


def grow(J, N, beta):

    seed_x, seed_y = random.randint(1, N - 2), random.randint(1, N - 2)

    # lattice = initialize(N)
    lattice = np.ones((N + 2, N + 2))
    lattice[N, N] = -1

    flipped = np.ones(lattice.shape) * -1
    spin0 = lattice[seed_y, seed_x]

    lattice = lattice * spin0  # for visualisation (1: spin0)
    # visualize(N, lattice, f"init_{N}") #TODO: uncomment

    cluster_list = np.ones((N * N, 2))
    cluster_list[:, 0], cluster_list[:, 1] = seed_y, seed_x
    cluster_list[0, :] = seed_y, seed_x

    cluster_list, cluster_count = wolff_seq(
        N, spin0, seed_x, seed_y, beta, J, lattice, cluster_list
    )
    lattice, flipped = flip(lattice, cluster_list, seed_x, seed_y, flipped)

    print(f"flippex {flipped.shape}")

    # total_member = cluster_count

    # print(cluster_list[:total_member])

    visualize(N, lattice, f"flipped_{N}")
    visualize(N, flipped, f"crystal_{N}")

    for iter in range(10):

        outer_list = get_index_outer(cluster_list[:cluster_count])

        r = random.randint(0, len(outer_list) - 1)

        outer_idx = outer_list[r]

        seed_y, seed_x = cluster_list[int(outer_idx), :]

        cluster_list, cluster_count = wolff_seq(
            N, spin0, seed_x, seed_y, beta, J, lattice, cluster_list
        )
        # print(cluster_list)
        lattice, flipped = flip(lattice, cluster_list, seed_x, seed_y, flipped)

        # total_member += cluster_count

        visualize(N, lattice, f"flipped_{N}")
        visualize(N, flipped, f"crystal_{N}")
        # print(cluster_list[:total_member])

        # time.sleep(0.2)


def match(array_1, array_2):

    for i in array_1:
        x, y = i[0], i[1]

        for i in range(len(array_2)):
            if x == array_2[i, 0]:
                if y == array_2[i, 1]:

                    return True
