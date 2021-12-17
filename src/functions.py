import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def print_real_lattice(N, lattice, printf=True, Word="Real lattice"):

    if printf:
        print(f"{Word}: \n{lattice[1:N+1, 1:N+1]}")

    return lattice[1 : N + 1, 1 : N + 1]


def initialize(N, seed_x, seed_y):
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

    spin0 = lattice[seed_y, seed_x]

    lattice = lattice * spin0  # for visualisation (1: spin0)

    flipped = np.ones(lattice.shape) * -1

    return lattice, flipped


def check_element(x, y, list):

    for i in range(len(list)):
        if x == list[i, 0]:
            if y == list[i, 1]:

                return True


"""
test = np.array([1,2])
test_2 = np.ones((1,2))
test_2[0,:] = 1, 2

print(test_check(1,2,test_2))

test = np.array([[4,2], [1,23]])
print(check_element(1,2,test))
"""


def get_P_add(beta, J):

    return 1 - np.exp(-2 * beta * J)


def is_neighbour(cy, cx, crystal):

    dxy = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])

    sum = 0

    for j in dxy:

        nx, ny = cx + j[1], cy + j[0]

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


def not_terminate(lattice, cluster_list):

    lattice[-1, :], lattice[:, -1], lattice[0, :], lattice[:, 0] = 0, 0, 0, 0

    outer_list = get_outer_list(cluster_list)

    dxy = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])

    sum = 0

    for i in outer_list:

        y, x = i[0], i[1]

        sum += lattice[int(y), int(x)]

        for j in dxy:

            cy, cx = y + j[0], x + j[1]

            sum += lattice[int(cy), int(cx)]

    if sum > 0:
        return True
    else:
        return False


def visualize(N, lattice, name, folder="record", printf=False):

    lattice = print_real_lattice(N, lattice, printf=printf, Word=name)

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


def make_GIF(N, beta, J, foldername="record", clean=True):

    Path(f"{foldername}/{N}/{beta}").mkdir(parents=True, exist_ok=True)

    import imageio

    iterations = 0

    for path in Path(f"{foldername}/{N}/animate/combined").iterdir():
        if path.is_file():
            iterations += 1

    filename = [
        f"{foldername}/{N}/animate/combined/{idx}.png" for idx in range(iterations)
    ]

    with imageio.get_writer(
        f"{foldername}/{N}/{beta}/iter{iterations-1}.gif", mode="I"
    ) as writer:

        for name in filename:

            image = imageio.imread(name)
            writer.append_data(image)

    print(f"animation saved as {foldername}/{N}/{beta}/iter{iterations-1}.gif")

    if clean:

        shutil.rmtree(f"{foldername}/{N}/animate")


def animate(N, lattice, flipped, previous_count, iterations):

    idx = previous_count + iterations

    save_frame(N, lattice, "flipped", iteration=idx)
    save_frame(N, flipped, "crystal", iteration=idx)
    combine(N, iteration=idx)


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

            cx, cy = x + j[1], y + j[0]

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


def get_neighbour_list(cy, cx):

    dxy = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])

    neighbour = np.zeros((4, 2))

    for i in range(4):
        neighbour[i, 0], neighbour[i, 0] = cy + dxy[i, 0], cx + dxy[i, 1]

    return neighbour


def get_seed(N):

    seed_x, seed_y = random.randint(1, N - 2), random.randint(1, N - 2)

    return seed_x, seed_y


def preparation(lattice, seed_x, seed_y):

    """Global array"""

    cluster_list = np.ones((1, 2))
    cluster_list[0, :] = seed_y, seed_x

    crystal = np.zeros(lattice.shape)  # for neighbour
    crystal[seed_y, seed_x] = 1

    return cluster_list, crystal


def get_seed2b_list(N, lattice):

    not_flipped = np.zeros((N * N, 2))

    count = 0

    for r in range(1, N + 1):
        for c in range(1, N + 1):

            spinc = lattice[r, c]

            if spinc > 0:

                not_flipped[count, :] = r, c

                count += 1

    return not_flipped[:count]


"""
test = np.array([[1,2], [3,4]])
test2 = np.array([[2,3],[3,4],[5,6]])
print(match(test, test2))
"""
