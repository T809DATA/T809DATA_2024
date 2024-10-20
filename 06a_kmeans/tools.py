import itertools
import sklearn.datasets as datasets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg


def load_iris():
    '''
    Load the iris dataset that contains N input features
    of dimension F and N target classes.

    Returns:
    * inputs (np.ndarray): A [N x F] array of input features
    * targets (np.ndarray): A [N,] array of target classes
    '''
    iris = datasets.load_iris()
    return iris.data, iris.target, [0, 1, 2]


def image_to_numpy(path: str = './images/buoys.png'):
    '''Converts an image to a numpy array and returns
    it and the original width and height
    '''
    # we only want the RGB values, not the A value.
    image = plt.imread(path)[:, :, :3]
    # reshape image from [width, height, 3] to [width*height, 3]
    return image.reshape(-1, image.shape[2]), (image.shape[0], image.shape[1])


def plot_gmm_results(
    X: np.ndarray,
    prediction: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray
):
    '''
    Plots all samples in X in a 2-D space where the
    color of samples is the same if they have the same
    prediction. Additionally, the gaussian distributions
    described by the means and covariances matrices are
    plotted in the corresponding colors.

    Input arguments:
    * X (np.ndarray): A [n x f] array of features.
    * prediction (np.ndarray): A [n] array which is the result
        of calling classifier.predict(X) where classifier
        is any sklearn classifier.
    * means (np.ndarray): A [k x f] array of mean vectors
    * covariances (np.ndarray): A [k x f x f] array of
        covariance matrices.
    '''
    color_iter = itertools.cycle(
        ['steelblue', 'mediumpurple', 'plum', 'gold', 'pink'])
    splot = plt.subplot(1, 1, 1)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(prediction == i):
            continue
        plt.scatter(
            X[prediction == i, 0], X[prediction == i, 1], .8, color=color)
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = matplotlib.patches.Ellipse(
            mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title('Gaussian Mixture')
    plt.show()


if __name__ == '__main__':
    print(image_to_numpy().shape)
