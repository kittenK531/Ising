import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def print_real_lattice(N, lattice, printf=True, Word="Real lattice"):

    if printf:
        print(f"{Word}: \n{lattice[1:N+1, 1:N+1]}")

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
    print_real_lattice(N, lattice, Word="Initialized lattice")

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
test = np.array([1,2])
test_2 = np.ones((1,2))
test_2[0,:] = 1, 2

print(check_element(1,2,test_2))

test = np.array([[4,2], [1,23]])
print(check_element(1,2,test))
"""


def get_P_add(beta, J):

    return 1 - np.exp(-2 * beta * J)


def is_neighbour(cy, cx, crystal):

    dxy = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])

    sum = 0

    for j in dxy:

        nx = cx + j[1]
        ny = cy + j[0]

        sum += crystal[ny, nx]

    if sum > 0:
        return True
    else:
        return False


def not_outer(x, y, cluster_list):

    dxy = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])

    Flag = True

    for j in dxy:

        cx = x + j[1]
        cy = y + j[0]

        Flag = Flag and check_element(cx, cy, cluster_list)

    return Flag


def get_index_outer(cluster_list):

    out_index = np.zeros(len(cluster_list))
    out_count = 0

    for idx in range(len(cluster_list)):

        y, x = cluster_list[idx, :]

        if not not_outer(x, y, cluster_list):

            out_index[out_count] = idx
            out_count += 1

            # print(f"outer: {cluster_list[idx]}")

    if out_count == 1:
        return out_index
    else:
        return out_index[:out_count]


def get_outer_list(cluster_list):

    index_array = get_index_outer(cluster_list)
    num_outer = len(index_array)
    outer_list = np.zeros((num_outer, 2))

    for i in range(num_outer):
        outer_list[i] = cluster_list[int(index_array[i]), :]

    return outer_list


def visualize(N, lattice, name, folder="record"):

    lattice = print_real_lattice(N, lattice, printf=False, Word=name)

    for r in range(N):
        for c in range(N):
            if lattice[r, c] < 0:
                lattice[r, c] = 0

    Path(f"{folder}/{N}").mkdir(parents=True, exist_ok=True)

    plt.imshow(lattice, cmap="Greys")
    plt.savefig(f"{folder}/{N}/{name}_{N}.png")
    plt.show(block=False)


def save_frame(N, lattice, name, iteration, folder="record"):

    lattice = print_real_lattice(N, lattice, printf=False, Word=name)

    Path(f"{folder}/{N}/animate").mkdir(parents=True, exist_ok=True)

    for r in range(N):
        for c in range(N):
            if lattice[r, c] < 0:
                lattice[r, c] = 0

    plt.imshow(lattice, cmap="Greys")
    plt.savefig(f"{folder}/{N}/animate/{name}_{iteration}.png")
    plt.show(block=False)


def combine(N, iteration, foldername="record", name_1="flipped", name_2="crystal"):

    from PIL import Image

    im1 = Image.open(f"{foldername}/{N}/animate/{name_1}_{iteration}.png")
    im2 = Image.open(f"{foldername}/{N}/animate/{name_2}_{iteration}.png")

    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))

    Path(f"{foldername}/{N}/animate/combined").mkdir(parents=True, exist_ok=True)
    dst.save(f"{foldername}/{N}/animate/combined/{iteration}.png")


def make_GIF(N, iterations, foldername="record", clean=True):

    import imageio

    filename = [
        f"{foldername}/{N}/animate/combined/{idx}.png" for idx in range(iterations + 1)
    ]

    with imageio.get_writer(
        f"{foldername}/{N}/grow_{iterations}.gif", mode="I"
    ) as writer:

        for name in filename:

            image = imageio.imread(name)
            writer.append_data(image)

    if clean:

        shutil.rmtree(f"{foldername}/{N}/animate")


def sequence_loop(N, start_y, start_x):

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

    print_real_lattice(N, array, printf=False, Word="Order")

    return visited_list


def flip(lattice, cluster_list, seed_x, seed_y, flipped):

    for index in cluster_list:
        r, c = index[0], index[1]

        lattice[int(r), int(c)] = -1 * lattice[int(r), int(c)]
        flipped[int(r), int(c)] = 1

    lattice[int(seed_y), int(seed_x)] = -1

    return lattice, flipped


""" back up """


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


""" Debug mode """


def debug_mode(J, N, beta):

    seed_x, seed_y = random.randint(1, N - 2), random.randint(1, N - 2)

    lattice = np.ones((N + 2, N + 2))
    lattice[N, N] = -1

    flipped = np.ones(lattice.shape) * -1
    spin0 = lattice[seed_y, seed_x]

    lattice = lattice * spin0  # for visualisation (1: spin0)
    visualize(N, lattice, f"init_{N}")  # TODO: uncomment

    cluster_list = np.ones((N * N, 2))
    cluster_list[:, 0], cluster_list[:, 1] = seed_y, seed_x
    cluster_list[0, :] = seed_y, seed_x

    cluster_list, cluster_count = wolff_seq(
        N, spin0, seed_x, seed_y, beta, J, lattice, cluster_list
    )
    lattice, flipped = flip(N, lattice, cluster_list, seed_x, seed_y, flipped)

    print(f"flippex {flipped.shape}")

    visualize(N, lattice, f"flipped_{N}")
    visualize(N, flipped, f"crystal_{N}")

    return seed_x, seed_y


# grow(1, 10, 0.3)


def get_neighbour_list(cy, cx):

    dxy = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])

    neighbour = np.zeros((4, 2))

    for i in range(4):
        neighbour[i, 0], neighbour[i, 0] = cy + dxy[i, 0], cx + dxy[i, 1]

    return neighbour


def match(array_1, array_2):

    for i in array_1:
        x, y = i[0], i[1]

        for i in range(len(array_2)):
            if x == array_2[i, 0]:
                if y == array_2[i, 1]:

                    return True


"""
test = np.array([[1,2], [3,4]])
test2 = np.array([[2,3],[3,4],[5,6]])
print(match(test, test2))
"""
