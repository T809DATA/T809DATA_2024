from typing import Union

import torch
import sklearn.datasets as datasets


def load_regression_iris():
    '''
    Load the regression iris dataset that contains N
    input features of dimension F-1 and N target values.

    Returns:
    * features (np.ndarray): A [N x F-1] array of input features
    * targets (np.ndarray): A [N,] array of target values
    '''
    iris = datasets.load_iris()
    return torch.tensor(iris.data[:, 0:3]).float(), torch.tensor(iris.data[:, 3]).float()


def split_train_test(
    features: torch.Tensor,
    targets: torch.Tensor,
    train_ratio: float = 0.8
) -> Union[tuple, tuple]:
    '''
    Shuffle the features and targets in unison and return
    two tuples of datasets, first being the training set,
    where the number of items in the training set is according
    to the given train_ratio
    '''
    p = torch.random.permutation(features.shape[0])
    features = features[p]
    targets = targets[p]

    split_index = int(features.shape[0] * train_ratio)

    train_features, train_targets = features[0:split_index, :],\
        targets[0:split_index]
    test_features, test_targets = features[split_index:, :],\
        targets[split_index:]

    return (train_features, train_targets), (test_features, test_targets)
