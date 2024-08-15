# Introduction Assignment
This is an introductory assignment for the course. By solving this assignment you will get more familiar with the tools we will be using in the course.

**Solving this assignment is not a requirement for participation in the course.**

## Setting up
Before starting, make sure that you have installed Python requirements as explained in the [main README](../README.md).

All your code should be turned in as a python script, i.e.
* A single file with a `.py` ending. Although you can use the `template.ipynb` notebook in your development and then port your code over to `template.py` if that suits you.
* The file should contain all the functions with **exactly** the same function names as listed in the assignment description below
* Use the supplied `template.py` file to fill in your code
* Plots should be turned in as well. The naming convention for submitted plots is as follows: The Z-th plot under Part X.Y should be turned in as `X_Y_Z.png`
* Store your plots in a seperate directory called `plots`
* Turn these files in as a `.zip` file.

## Part 1
### 1.1
Define the Gaussian probability density function `normal_prob(x, sigma, mu)` where:
    * $\sigma$ is the standard deviation
    * $\mu$ is the mean of the distribution
    * $x$ is any real value

The Gaussian PDF is given by

$$p(x) = \frac{1}{\sqrt{ 2 \pi \sigma^2 }}e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} }$$

$p(x)$ will give you the probability of the value $x$ according to the Gaussian PDF.

Make sure to use `numpy` math operators, for example:
* for $x^2$ do: `np.power(x, 2)`
* for $\pi$ do: `np.pi`
* for $e^2$ do: `np.exp(2)`
* For multiplication/division you can simply use `*` and `/` respectively.
If done correctly, you can even supply $x$ as an $n$-dimensional numpy array and you will get $n$ probabilities as a result.

Example inputs and outputs:
* `normal(0, 1, 0) = 0.3989422804014327`
* `normal(3, 1, 5) = 0.05399096651318806`
* `normal(np.array([-1,0,1]), 1, 0) = array([0.24197072, 0.39894228, 0.24197072])`

### 1.2
Define a function `plot_normal(sigma, mu, x_start, x_end)` that plots a normal distribution on the given range, [x_start, x_end]. This function should work similarily to `examples.plot_multiple` in that it should only plot onto a figure without actually showing it. Inside the function, create 500 linear spaced values using `np.linspace`.

Calling `plot_normal(0.5, 0, -2, 2)` should result in the following plot:

![Simple Normal Distribution Plot](images/one_normal.png)

Now plot the three following normal distributions onto the same figure on the range $[-5, 5]$:
* $\mu = 0, \sigma=0.5$
* $\mu = 1, \sigma=0.25$
* $\mu = 1.5, \sigma=1$

Save the figure as `1_2_1.png`

## Part 2

### 2.1
Create a function `normal_mixture(x, sigmas, mus, weights)` which defines a Gaussian mixture function. A Gaussian mixture density function combines multiple gaussian densities with weights that sum to 1. It is given by:

$$p(x) = \sum_{i=1}\frac{\phi_i}{\sqrt{ 2 \pi \sigma_i^2 }}\text{exp}\left({ - \frac{ (x - \mu_i)^2 } {2 \sigma_i^2} }\right)$$

where $\phi_i$ is the $i$-th mixture weight, corresponding to the $i$-th gaussian.

Assume that the `x` argument is always a `np.ndarray` of some sort.

Example inputs and outputs:

Example 1:

```
normal_mixture(np.linspace(-5, 5, 5), [0.5, 0.25, 1], [0, 1, 1.5], [1/3, 1/3, 1/3])
```
Output: `[8.89852205e-11 4.56012216e-05 3.09312492e-01 8.06579074e-02 2.90894232e-04]`

Example 2:

```
normal_mixture(np.linspace(-2, 2, 4), [0.5], [0], [1])
```
Output: `[2.67660452e-04 3.28020149e-01 3.28020149e-01 2.67660452e-04]`

### 2.2
Lets compare three individual normal components with the evenly-weighted mixture of the three. You should use your `plot_normal` function for the individual components but a different method must be used for the mixture.

Try the following values:
* $\mu = 0, \sigma=0.5, \phi=1/3$
* $\mu = -0.5, \sigma=1.5, \phi=1/3$
* $\mu = 1.5, \sigma=0.25, \phi=1/3$

Plot each individual normal distribution as well as the mixture onto the same figure. Save this figure as `2_2_1.png`

## Part 3
### 3.1
Define a function `sample_gaussian_mixture(sigmas, mus, weights, n_samples)` that samples `n_samples` from a given gaussian mixture. To sample a Gaussian mixture it is best to do the following:

1. Use `np.random.multinomial` to determine how many times each normal component is selected to sample from. For example, to throw a dice 10 times using this function you would do: `np.random.multinomial(10, [1/6.]*6)` since each side of the dice is equally likely, i.e. $1/6$. In your case, these probabilities are determined by the weights in your mixture
2. Use `np.random.normal` to sample the selected normal component
3. Repeat step 2 until you have `n_samples`


This function should return a `numpy.ndarray`

Example inputs and outputs (Use np.random.seed(0) to get similar values):

Example 1:

```
sample_gaussian_mixture([0.1, 1], [-1, 1], [0.9, 0.1], 3)
= [-0.95352122 -0.88278249 -1.11422348]
```

Example 2:
```
[0.1, 1, 1.5], [1, -1, 5], [0.1, 0.1, 0.8], 10)
= [ 0.91172091 -0.83706012  3.91024544  3.98949614  6.1565694   5.81700712
  3.79369706  4.36246244  6.70470375  5.17662141]
```

### 3.2
Lets now see how well samples follow a specific mixture of normal probability distributions.

Our mixture will have the following parameters:
* $\mu = 0, \sigma=0.3, \phi=0.2$
* $\mu = -1, \sigma=0.5, \phi=0.3$
* $\mu = 1.5, \sigma=1, \phi=0.5$

After drawing:
1. 10 samples
2. 100 samples
3. 500 samples
4. 1000 samples

Plot a histogram of the samples, using e.g `plt.hist(samples, 100, density=True)`, as well as the gaussian mixture.

You should use `plt.subplot` beteween each experiment, i.e. just before you draw the first plot, do: `plt.subplot(141)`, and just before the second do: `plt.subplot(142)`

For a different mixture, the plot looks like this after 100 draws:

![Mixtures and samples](images/mixture_and_samples.png)

Save this plot as `3_2_1.png`

