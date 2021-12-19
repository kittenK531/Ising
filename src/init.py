import random
import time
from datetime import datetime

import numpy as np

from functions import (
    flip,
    get_index_outer,
    get_P_add,
    get_seed,
    get_seed2b_list,
    initialize,
    is_neighbour,
    not_terminate,
    preparation,
    print_real_lattice,
    sequence_loop,
    visualize,
)

current_time = datetime.now()


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
    beta,
    start_x,
    start_y,
    Pr,
    cluster_list,
    flipped,
    lattice_pos,
    crystal,
    previous_count,
    detailed=False,
):

    seed_y, seed_x = start_y, start_x
    lattice = lattice_pos

    spin0 = lattice[seed_y, seed_x]

    i = 0

    while not_terminate(lattice, cluster_list):
        array = sequence_loop(N, seed_y, seed_x)

        start = time.time()
        cluster_list = growth(array, lattice, spin0, Pr, cluster_list, crystal)
        end = time.time()

        print(f"small iteration time {end - start}")

        start_f = time.time()
        lattice, flipped = flip(lattice, cluster_list, seed_x, seed_y, flipped)
        end_f = time.time()
        print(f"flip time {end_f - start_f}")

        index_arr = get_index_outer(cluster_list)
        # print(index_arr)
        r = random.randint(0, len(index_arr) - 1)
        seed_y, seed_x = cluster_list[r, :]
        print(f"beta: {beta:.2f} iter: {previous_count + i + 1} Seed: {seed_y, seed_x}")
        i += 1

    iterations = i

    return lattice, flipped, iterations


def whole_growth(N, beta, J, detailed=False):

    start_i = time.time()
    seed_y, seed_x = get_seed(N)

    lattice, flipped = initialize(N, seed_x, seed_y)

    previous_count = 0

    # animate(N, beta, lattice, flipped, previous_count, 0)

    i = 0

    visualize(N, beta, lattice, f"init")

    Pr = get_P_add(beta, J)

    end_i = time.time()
    print(f"initialize time: {end_i - start_i}")

    real_lattice = print_real_lattice(N, lattice, False)

    while real_lattice.sum() > 0:

        # print(f"Overall cluster iteration: {i+1}")

        start_c = time.time()
        cluster_list, crystal = preparation(lattice, seed_x, seed_y)
        end_c = time.time()

        print(f"cluster iter time: {end_c - start_c}")

        lattice, flipped, cluster_iter = iterative(
            N,
            beta,
            seed_x,
            seed_y,
            Pr,
            cluster_list,
            flipped,
            lattice,
            crystal,
            previous_count,
            detailed,
        )
        """
        if not detailed:
            animate(N, beta, lattice, flipped, 0, iterations=i + 1)
        """
        unflipped = get_seed2b_list(N, lattice)

        previous_count += cluster_iter

        if len(unflipped) - 1 >= 0:
            r = random.randint(0, len(unflipped) - 1)

            seed_y, seed_x = int(unflipped[r, 0]), int(unflipped[r, 1])

        else:
            break

        i += 1

        real_lattice = print_real_lattice(N, lattice, False)

    print(f"beta: {beta:.2f} Total count of iterations: {previous_count}")
    visualize(
        N,
        beta,
        lattice,
        f"flipped_{previous_count}_{current_time.day}{current_time.hour}{current_time.minute}",
    )
    visualize(
        N,
        beta,
        flipped,
        f"crystal_{previous_count}_{current_time.day}{current_time.hour}{current_time.minute}",
    )

    # make_GIF(N, beta, J, total = previous_count, clean=True)


""" Execution """

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--N", default=5, type=int)
parser.add_argument("--beta", default=0.2, type=float)
parser.add_argument("--J", default=1.0, type=float)
parser.add_argument("--detailed", default=False, type=bool)

args = parser.parse_args()

whole_growth(args.N, args.beta, args.J, detailed=args.detailed)

"""
def main():
    with Pool() as pool:
        pool.starmap(
            whole_growth,
            [
                (args.N, args.beta, args.J),
                (args.N, args.beta + 0.1, args.J),
                (args.N, args.beta + 0.2, args.J),
                (args.N, args.beta + 0.3, args.J),
                (args.N, args.beta + 0.4, args.J),
            ],
        )


if __name__ == "__main__":
    freeze_support()
    main()
"""

""" python3 init.py --N 5 --beta 0.2 """
