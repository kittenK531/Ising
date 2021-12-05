import random

import numpy as np
from tqdm import tqdm


def print_real_lattice(N, lattice, islattice=True):

    if islattice:
        print(f"Real lattice: \n{lattice[1:N+1, 1:N+1]}")
    else:
        print(f"Order: \n{lattice[1:N+1, 1:N+1]}")

    return lattice[1 : N + 1, 1 : N + 1]


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

    # print(f"real lattice:\n {lattice}")

    print(f"The initialized random configuration of {N}*{N} lattice is")
    print_real_lattice(N, lattice)

    """ Periodic Boundary condition """
    lattice[-1, :] = lattice[1, :]
    lattice[:, -1] = lattice[:, 1]
    lattice[0, :] = lattice[N - 1, :]
    lattice[:, 0] = lattice[:, N - 1]

    # print(f"After boundary condition, the lattice becomes:\n {lattice}")

    return lattice


def check_element(x, y, list):

    for i in range(len(list)):
        if x == list[i, 0]:
            if y == list[i, 1]:

                return True


"""
test = np.array([[4,2], [1,23]])
print(check_element(1,2,test))
"""


def get_P_add(beta, J):

    return 1 - np.exp(-2 * beta * J)


def wolff_seq(N, spin0, seed_x, seed_y, beta, J, lattice):

    # enter the start of x, y in x+1, y+1 in manual use

    P_add = get_P_add(beta, J)
    # array = np.zeros((N + 2, N + 2)) # Order

    iteration_count, current_count, visit_number, cluster_count = 1, 1, 0, 0

    start_x, start_y = seed_x, seed_y

    visited_list = np.ones((N * N, 2))
    visited_list[:, 0], visited_list[:, 1] = start_y, start_x

    cluster_list = np.ones((N * N, 2))
    cluster_list[:, 0], cluster_list[:, 1] = start_y, start_x
    cluster_list[0, :] = seed_y, seed_x

    x, y = start_x, start_y
    cx, cy = x, y

    cluster_x, cluster_y = cx, cy

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
                and check_element(y, x, cluster_list)
            ):

                cspin = lattice[int(cy), int(cx)]

                # print(f"site = {cy, cx} align = {(spin0 * cspin > 0)}, probability pass = {(random.uniform(0,1) < P_add)}")

                if (spin0 * cspin > 0) and (random.uniform(0, 1) < P_add):
                    cluster_list[current_count, :] = int(cy), int(cx)
                    cluster_x, cluster_y = int(cx), int(cy)
                    Flag = True
                    cluster_count += 1
                else:
                    Flag = False

                # array[int(cy), int(cx)] = current_count # order
                current_count += 1
                visited_list[current_count - 1, :] = int(cy), int(cx)

            iteration_count += 1

        visit_number += 1
        y, x = visited_list[visit_number, :]  # order
        print(
            f"previous cluster {cluster_y, cluster_x}\tThis point {cy, cx} {Flag}\tThe next center is {y,x}"
        )
        # y, x = cluster_list[cluster_count]

    # print_real_lattice(N, array, islattice=False)

    return cluster_list


def flip(N, lattice, cluster_list, seed_x, seed_y):

    flipped = np.ones((N + 2, N + 2)) * -1

    for index in cluster_list:
        r, c = index[0], index[1]

        lattice[int(r), int(c)] = -1 * lattice[int(r), int(c)]
        flipped[int(r), int(c)] = 1

    lattice[seed_y, seed_x] = -1

    return lattice, flipped


def visualize(N, lattice, name):

    lattice = print_real_lattice(N, lattice)

    for r in range(N):
        for c in range(N):
            if lattice[r, c] < 0:
                lattice[r, c] = 0

    print(lattice)

    import matplotlib.pyplot as plt

    plt.imshow(lattice, cmap="Greys")
    plt.savefig(f"{name}.png")
    plt.show(block=False)


""" Execution """
J = 1.0
N = 5
beta = 1.12

seed_x, seed_y = random.randint(1, N - 2), random.randint(1, N - 2)

lattice = initialize(N)
spin0 = lattice[seed_y, seed_x]

lattice = lattice * spin0  # for visualisation (1: spin0)
visualize(N, lattice, "init")


# print(get_P_add(beta, J))
cluster_list = wolff_seq(N, spin0, seed_x, seed_y, beta, J, lattice)
# print(cluster_list)
lattice, flipped = flip(N, lattice, cluster_list, seed_x, seed_y)
visualize(N, lattice, "flipped")
visualize(N, flipped, "crystal")
