import random

import numpy as np

from functions import (
    get_seed,
    initialize,
    make_GIF_local,
    not_converge,
    preparation,
    save_frame,
    sequence_loop,
    visualize,
)


def tabulated_energy(beta, J=1.0):

    energy_record = np.zeros((5, 2))

    for idx, dEo in enumerate([-8, -4, 0, 4, 8]):
        energy_record[idx, :] = dEo, np.clip(np.exp(-1 * beta * dEo * J), 0, 1)

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


def MCS(N, beta, J, lattice, seed_y, seed_x, cluster_list, flipped, count):

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

        count += 1

    """
    print(f"Energy difference is {E_diff}")
    print(f"The boltzmann factor is {get_Boltzmann_factor(E_diff, energy2table)}.")
    """

    return lattice, cluster_list, flipped, count


def iterative(
    N, beta, J, lattice, seed_y, seed_x, cluster_list, flipped, detailed=True
):

    x, y = seed_y, seed_x

    array = sequence_loop(N, x, y)

    count = 0

    for i in array:

        x, y = int(i[0]), int(i[1])

        lattice, cluster_list, flipped, count = MCS(
            N, beta, J, lattice, x, y, cluster_list, flipped, count
        )

        print(f"iter: {count + 1} Seed: {y, x}")

        if detailed:
            save_frame(
                N, beta, lattice, "flipped", iteration=count + 1, folder="record_local"
            )

    return lattice, flipped, count


def run(N, beta, J, detailed=False):

    seed_y, seed_x = get_seed(N)

    lattice, flipped = initialize(N, seed_x, seed_y)

    visualize(N, beta, lattice, f"init", folder="record_local")

    save_frame(N, beta, lattice, "flipped", iteration=0, folder="record_local")

    i = 0

    previous_count = 0

    while not_converge(N, lattice):

        print(f"Overall MCS teration: {i+1}")

        cluster_list, crystal = preparation(lattice, seed_x, seed_y)

        lattice, flipped, count = iterative(
            N, beta, J, lattice, seed_y, seed_x, cluster_list, flipped
        )

        visualize(N, beta, lattice, "flipped", folder="record_local")

        if not detailed:
            save_frame(
                N, beta, lattice, "flipped", iteration=i + 1, folder="record_local"
            )

        seed_y, seed_x = get_seed(N)

        i += 1

        previous_count += count

    make_GIF_local(N, beta, J, previous_count, "flipped", clean=True)


# Note: b = 0.2 converge on 3rd run

""" Execution """

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--N", default=5, type=int)
parser.add_argument("--beta", default=0.2, type=float)
parser.add_argument("--J", default=1.0, type=float)
parser.add_argument("--detailed", default=False, type=bool)

args = parser.parse_args()

for dbeta in np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.12, 1.13, 1.2]
):
    run(args.N, dbeta, args.J, detailed=False)

"""
from multiprocessing import Pool, freeze_support

def main():
    with Pool() as pool:
        pool.starmap(run, [(args.N, args.beta, args.J), (args.N, args.beta + 0.1, args.J), (args.N, args.beta + 0.2, args.J), (args.N, args.beta + 0.3, args.J)])

if __name__=="__main__":
    freeze_support()
    main()
"""
""" python3 MCS.py --N 50 --beta 0.2 """
