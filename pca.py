from re import M, U
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    # TODO: add your code here
    x = np.load(filename)
    x = x - np.mean(x, axis=0)
    return x


def get_covariance(dataset):
    # TODO: add your code here
    n = len(dataset)
    data = np.zeros((1024, 1024))
    temp = np.zeros((1024, 1024))
    for pic in dataset:
        mat = pic[np.newaxis]
        temp = np.dot(np.transpose(mat), mat)
        data = np.add(data, temp)
    data = data/(n-1)
    return data


def get_eig(S, m):
    # TODO: add your code here
    T_Lambda, T_U = eigh(S)
    n = len(T_Lambda)
    L, U = eigh(S, subset_by_index=[n - m, n - 1])
    Matrix1 = np.zeros((len(L), len(L)))
    rL = np.flip(L)

    for i in range(len(L)):
        Matrix1[i][i] = rL[i]

    for i in range(len(U)):
        U[i] = np.flip(U[i])

    return Matrix1, U


def get_eig_perc(S, perc):
    # TODO: add your code here
    T_Lambda, T_U = eigh(S)
    Limit = np.sum(T_Lambda) * perc
    L, U = eigh(S, subset_by_value=[Limit, np.inf])

    Matrix1 = np.zeros((len(L), len(L)))
    rL = np.flip(L)

    for i in range(len(L)):
        Matrix1[i][i] = rL[i]

    for i in range(len(U)):
        U[i] = np.flip(U[i])

    return Matrix1, U


def project_image(img, U):
    # TODO: add your code
    count = np.zeros((len(U)))
    temp = 0
    for i in range(0, len(U[0])):
        column = U[:, i]
        temp = np.dot(np.transpose(column), img)
        count += temp * column
    return count


def display_image(orig, proj):
    # TODO: add your code here
    o_img = np.reshape(orig, (32, 32))
    p_img = np.reshape(proj, (32, 32))
    o_img = np.transpose(o_img)
    p_img = np.transpose(p_img)
    fig, axs = plt.subplots(1, 2)
    axs[0].set_title('Original')
    axs[1].set_title('Projection')
    o_c_bar = axs[0].imshow(o_img, aspect='equal')
    p_c_bar = axs[1].imshow(p_img, aspect='equal')
    fig.colorbar(o_c_bar, ax=axs[0])
    fig.colorbar(o_c_bar, ax=axs[1])
    plt.show()
