import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def estimate_covariance():
    values = np.array([
        [0.30, 8.60],
        [0.20, 5.60],
        [0.15, 2.30],
        [0.35, 9.90],
        [0.05, 0.10],
    ])
    # 'It seems like these values are correlated, lets plot.
    plt.scatter(values[:, 0], values[:, 1])
    plt.show()
    # We can see from the plot that the values are indeed
    # somewhat correlated. Lets print the covariance matrix
    print(np.cov(values, rowvar=False))
    # The values that are off the diagonal axis are covariances
    # of each pair of variables in our data. A higher value
    # means that the variables are more correlated
    values = np.array([
        [0.30, 8.60],
        [0.20, 5.60],
        [5.15, 9.30],
        [0.35, 4.90],
        [9.05, 0.10],
    ])
    # These variables don't seem that correlated. Lets plot again.
    plt.scatter(values[:, 0], values[:, 1])
    plt.show()
    # The values are all over the place! Lets print the covariance
    # matrix
    print(np.cov(values, rowvar=False))
    # We are right again. The off-diagonal values are way lower now,
    # meaning that the variables are less correlated.


def multivariate_normal_pdf():
    # lets assume that we have estimated some mean and covariance
    # of some class in a dataset
    mean = np.array([0.5, 3.8])
    cov = np.array([
        [1.4250e-02, 4.8750e-01],
        [4.8750e-01, 1.7045e+01]])
    # Using these parameters we can define a multivariate normal
    # probability distribution. We can use the distribution to
    # acquire the relative likelihood of a new data point being
    # sampled from the distribution:
    print(multivariate_normal(mean=mean, cov=cov).pdf([0, 0]))
    print(multivariate_normal(mean=mean, cov=cov).pdf([1, 1]))
    print(multivariate_normal(mean=mean, cov=cov).pdf([0.4, 3.9]))
    print(multivariate_normal(mean=mean, cov=cov).pdf([100, 100]))
    # It seems like [0.4, 3.9] is the most likely to be sampled
    # from this distribution


if __name__ == '__main__':
    print('Running estimate_covariance: ')
    estimate_covariance()
    print('Running pdf: ')
    pdf()
