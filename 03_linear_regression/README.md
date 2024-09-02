# Linear Models for Regression

So far we have tried to *classify* flowers from the Iris dataset. Classification tasks assume that we have a set of *features* and a set of *class labels* that we would like our models to learn to predict.

Regression tasks don't assume categorical labels. In regression tasks we again have some set of *features* but the value we want to predict isn't a label anymore but some value from a continuous domain. Therefore the the output domain of the model is continuous.

You can for example imagine the quadratic equation

$$
    y = ax^2 + bx + c
$$

An example of a regression task might be to predict what $a, b, c$ are given a feature set of multiple $(x, y)$ values.

In this assignment we will be using a altered version of the Iris dataset were we have ditched the class label targets and replaced them with the last feature column (representing the pedal length of the flower). You can load this data with `tools.load_regression_iris`.

# Pytorch

In this assignment we are going to use Pytorch instead of Numpy to handle arrays and matrices.
Pytorch stores arrays and matrices in [Tensors](https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html).
In future assignments we will be using Tensors to create more complicated models including Neural Networks.

## Requirements

You will need the following requirements to finish this requirement.

- `torch`: https://pytorch.org/
- `matplotlib`: https://matplotlib.org/
- `scikit-learn`: https://scikit-learn.org/stable/index.html

You can run `pip install -r requirements.txt` to install all the requirements.

## Section 1
We are going to use the Gaussian basis functions to predict real valued target variables of our data.

$$
    \phi_k(x) = \frac{1}{(2\pi)^{D/2}} \frac{1}{|\Sigma_k|^{1/2}} e^{-\frac{1}{2} (x-\mu_k)^T \Sigma_k^{-1} (x-\mu_k)}
$$

We control the shift of the basis function using the mean vector $\mu_k$, but we force all covariance matrices to be identical and diagonal $\Sigma_k = \sigma\mathbf{I}$ for all $k$ so $\sigma$ is the parameter that controls the width of all the basis functions (in all directions).

Create a function `mvn_basis(features, mu, sigma)` that applies the multivariate normal basis function on the set of features. You should use the `scipy.stats.multivariate_normal` object to create your multivariate gaussians.

Example inputs and outputs:

*Load the data*
```
X, t = load_regression_iris()
N, D = X.shape
```

We need to define how many basis functions we are going to use and determine the mean vectors that we are using. We also have to define the variance, `sigma`. We do this arbitrarily.

```
M, sigma = 10, 10
mu = torch.zeros((M, D))
for i in range(D):
    mmin = torch.min(X[i, :])
    mmax = torch.max(X[i, :])
    mu[:, i] = torch.linspace(mmin, mmax, M)
fi = mvn_basis(X, mu, sigma)
```

*Output*
```
fi: tensor([[0.0008, 0.0010, 0.0012,  ..., 0.0014, 0.0012, 0.0011],
        [0.0010, 0.0012, 0.0013,  ..., 0.0013, 0.0012, 0.0010],
        [0.0010, 0.0012, 0.0014,  ..., 0.0013, 0.0012, 0.0010],
        ...,
        [0.0002, 0.0003, 0.0005,  ..., 0.0014, 0.0015, 0.0015],
        [0.0002, 0.0003, 0.0005,  ..., 0.0015, 0.0016, 0.0016],
        [0.0003, 0.0005, 0.0006,  ..., 0.0015, 0.0016, 0.0016]])
```

### Section 2

Plot the output of each basis function, using the same parameters as above, as a function of the features. You should plot all the outputs onto the same plot. Turn in your plot as `plot_1_2_1.png`.

A single basis function output is shown below

![Single basis function](images/single_basis.png)


### Section 3

Create a function `max_likelihood_linreg(fi, targets, lam)` that estimates the maximum likelihood values for the linear regression model.

Example inputs and outputs:
```
fi = mvn_basis(X, mu, sigma) # same as before
lamda = 0.001
wml = max_likelihood_linreg(fi, t, lamda)
```
*Output*:
```
wml: [  3.04661879   8.98364452  17.87352659  29.86144389  44.59162487
  61.12887842  78.01047234  93.44289728 105.61212037 113.03406275]
```

### Section 4
Finally, create a function `linear_model(features, mu, sigma, w)` that predicts targets given the weights from your `max_likelihood_linreg`, the basis functions, defined by `mu` and `sigma`, and the `features`.

Example inputs and outputs:

```
wml = max_likelihood_linreg(fi, t, lamda) # as before
prediction = linear_model(X, mu, sigma, wml)
```
*Output*:
```
prediction = [0.72174704 0.71462572 0.72048698 0.75382486 0.73161981 0.7494642
 0.7502493  0.74274217 0.73390818 0.73639504 0.71611075 0.77254612
 ...
 0.77988157 0.60555882 0.61733184 0.62902646 0.75304099 0.57770296
 0.61914859 0.64629346 0.67533913 0.67677248 0.72763469 0.76905279]
```

### Section 5
*This question should be answered in a raw text file, turn it in as `5_1.txt`*

- How good are these predictions?
- Use plots to show the prediction accuracy, either by plotting the actual values vs predicted values or the mean-square-error.

Submit your plots as `5_a.png`, `5_b.png`, etc. as needed.

### What to turn in to Gradescope
*Read this carefully before you submit your solution.*

You should edit `template.py` to include your own code.
 
This is an individual project, you can of course help each other out but your code should be your own.

You can use built-in python modules as you wish, however you are not allowed to install and import packages other than are already imported.

Files to turn in:

- `template.py`: This is your code
- `plot_1_2_1.png`
- `5_1.txt`
- `5_a.png`, `5_b.png`, ... as needed

Make sure the file names are exact. 
Submission that do not pass the first two tests in Gradescope will not be graded.


### Independent section (optional)
This is a completely open-ended independent section. You are free to test out the methods in the assignment on different datasets, change the models, change parameters, make visualizations, analyze the models, or anything you desire.

