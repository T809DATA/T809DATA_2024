import numpy as np
import matplotlib.pyplot as plt


def scatter_2d_data(data: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def scatter_3d_data(data: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def bar_per_axis(data: np.ndarray):
    for i in range(data.shape[1]):
        plt.subplot(1, data.shape[1], i+1)
        plt.hist(data[:, i], 100)
        plt.title(f'Dimension {i+1}')
    plt.show()
