# The Back-propagation Algorithm
The aim of this exercise is to implement the back-propagation algorithm. 
Pytorch has this functionality built into tensors already as `torch.autograd` (for more see [A Gentle Introduction to torch.autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)) but here we aim to implement 
the algorithm using only the linear algebra and other mathematics tools available in PyTorch. 
This way you get an insight into how the algorithm works before you move on to use it in larger toolboxes such as Pytorch.

We will restrict ourselves to fully-connected feed forward neural networks with one hidden layer (plus an input and an output layer).

## Requirements

You will need the following requirements to finish this requirement.

- `torch`: https://pytorch.org/
- `matplotlib`: https://matplotlib.org/
- `scikit-learn`: https://scikit-learn.org/stable/index.html

You can run `pip install -r requirements.txt` to install all the requirements.

!!!Note: If you are having trouble installing pytorch (or CUDA related dependencies) with the requirement file see [*Start Locally*](https://pytorch.org/get-started/locally/) on the PyTorch website and install the packages manually. F.ex `pip install torch torchvision torchaudio` and `pip install matplotlib scikit-learn`.

## Section 1

### Section 1.1 The Sigmoid function
We will use the following nonlinear activation function:

$$\sigma(a)=\frac{1}{1+e^{-a}}$$

We will also need the derivative of this function:

$$\frac{d}{da}\sigma(a) = \frac{e^{-a}}{(1+e^{-a})^2} = \sigma(a) (1-\sigma(a))$$

Create these two functions:
1. The sigmoid function: `sigmoid(x)`
2. The derivative of the sigmoid function: `d_sigmoid(x)`

**Note**: To avoid overflows, make sure that inside `sigmoid(x)` you check if `x<-100` and return `0.0` in that case.

Example inputs and outputs:
* `sigmoid(torch.Tensor([0.5]))` -> `tensor([0.6225])`
* `d_sigmoid(torch.Tensor([0.2]))` -> `tensor([0.2475])`

### Section 1.2 The Perceptron Function
A perceptron takes in an array of inputs $X$ and an array of the corresponding weights $W$ and returns the weighted sum of $X$ and $W$, 
as well as the result from the activation function (i.e. the sigmoid) of this weighted sum. *(see Eq. 6.10 & 6.11 in Bishop (Eq. 5.48 & 5.62 in old Bishop)*.

Implement the function `perceptron(x, w)` that returns the weighted sum and the output of the activation function.

Example inputs and outputs:
* `perceptron(torch.Tensor([1.0, 2.3, 1.9]), torch.Tensor([0.2, 0.3, 0.1]))` -> `(tensor(1.0800), tensor(0.7465))`
* `perceptron(torch.Tensor([0.2, 0.4]), torch.Tensor([0.1, 0.4]))` -> `(tensor(0.1800), tensor(0.5449))`

### Section 1.3 Forward Propagation
When we have the sigmoid and the perceptron function, we can start to implement the neural network.

Implement a function `ffnn` which computes the output and hidden layer variables for a single hidden layer feed-forward neural network. 
If the number of inputs is $D$, the number of hidden layer neurons is $M$ and the number of output neurons is $K$, 
the matrices $W_1$ of size $[(D+1)\times M]$ and $W_2$ of size $[(M+1)\times K]$ represent the linear transform from 
the input layer and the hidden layer and from the hidden layer to the output layer respectively.

Write a function `y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)` where:

* `x` is the input pattern of $(1\times D)$ dimensions (a line vector)
* `W1` is a $((D+1)\times M)$ matrix and `W2` is a $(M+1)\times K$ matrix. (the `+1` are for the bias weights)
* `a1` is the input vector of the hidden layer of size $(1\times M)$ (needed for backprop).
* `a2` is the input vector of the output layer of size $(1\times K)$ (needed for backprop).
* `z0` is the input pattern of size $(1\times (D+1))$, (this is just `x` with `1.0` inserted at the beginning to match the bias weight).
* `z1` is the output vector of the hidden layer of size $(1\times (M+1))$ (needed for backprop).
* `y` is the output of the neural network of size $(1\times K)$.

Example inputs and outputs:

*First load the iris data:*
```
# initialize the random generator to get repeatable results
torch.manual_seed(4321)
features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets) = \
    split_train_test(features, targets)
```

*Then call the function*
```
# initialize the random generator to get repeatable results
torch.manual_seed(1234)

# Take one point:
x = train_features[0, :]
K = 3  # number of classes
M = 10
D = 4
# Initialize two random weight matrices
W1 = 2 * torch.rand(D + 1, M) - 1
W2 = 2 * torch.rand(M + 1, K) - 1
y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
```

*Outputs*:
* `y` : `tensor([0.7079, 0.7418, 0.2414])`
* `z0`: `tensor([1.0000, 4.8000, 3.4000, 1.6000, 0.2000])`
* `z1`: `tensor([1.0000, 0.9510, 0.1610, 0.8073, 0.9600, 0.5419, 0.9879, 0.0967, 0.7041, 0.7999, 0.1531])`
* `a1`: `tensor([ 2.9661, -1.6510,  1.4329,  3.1787,  0.1681,  4.4007, -2.2343,  0.8669, 1.3854, -1.7101])`
* `a2`: `tensor([ 0.8851,  1.0554, -1.1449])`

### Section 1.4 Backward Propagation

We will now implement the back-propagation algorithm to evaluate the gradient of the error function $\Delta E_n(x)$.

Create a function `y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)` where:
* `x, M, K, W1` and `W2` are the same as for the `ffnn` function
* `target_y` is the target vector. In our case (i.e. for the classification of Iris) this will be a vector with 3 elements, with one element equal to 1.0 and the others equal to 0.0. (*).
* `y` is the output of the output layer (vector with 3 elements)
* `dE1` and `dE2` are the gradient error matrices that contain $\frac{\partial E_n}{\partial w_{ji}}$ for the first and second layers.

Assume sigmoid hidden and output activation functions and assume cross-entropy error function (for classification). 
Notice that $E_n(\mathbf{w})$ is defined as the error function for a single pattern $\mathbf{x}_n$. *The algorithm is described on page 237 in Bishop (page 244 in old Bishop)*.

The inner working of your `backprop` function should follow this order of actions:
1. run `ffnn` on the input.
2. calculate $\delta_k = y_k - target\\_y_k$
3. calculate: $\delta_j = \frac{d}{da} \sigma (a^1_j) \sum_{k} w_{k,j+1} \delta_k$ (the `+1` is because of the bias weights)
4. initialize `dE1` and `dE1` as zero-matrices with the same shape as `W1` and `W2`
5. calculate `dE1_{i,j}` $= \delta_j z^{(0)}_i$  and `dE2_{j,k}` = $\delta_k z^{(1)}_j$

Example inputs and outputs:

*Call the function*
```
# initialize random generator to get predictable results
torch.manual_seed(42)

K = 3  # number of classes
M = 6
D = train_features.shape[1]

x = features[0, :]

# create one-hot target for the feature
target_y = torch.zeros(K)
target_y[targets[0]] = 1.0

# Initialize two random weight matrices
W1 = 2 * torch.rand(D + 1, M) - 1
W2 = 2 * torch.rand(M + 1, K) - 1

y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
```
*Output*
* `y`: `tensor([0.7047, 0.5229, 0.5533])`
* `dE1`:
```
tensor([[ 3.0727e-02,  4.4601e-03,  4.3493e-04,  1.7125e-02,  9.1134e-04, -7.7194e-02],
        [ 1.5671e-01,  2.2747e-02,  2.2181e-03,  8.7336e-02,  4.6478e-03, -3.9369e-01],
        [ 1.0755e-01,  1.5611e-02,  1.5222e-03,  5.9936e-02,  3.1897e-03, -2.7018e-01],
        [ 4.3018e-02,  6.2442e-03,  6.0890e-04,  2.3974e-02,  1.2759e-03, -1.0807e-01],
        [ 6.1455e-03,  8.9203e-04,  8.6985e-05,  3.4249e-03,  1.8227e-04, -1.5439e-02]])
```
* `dE2`:
```
tensor([[-0.2953,  0.5229,  0.5533],
        [-0.1518,  0.2687,  0.2844],
        [-0.2923,  0.5175,  0.5476],
        [-0.2938,  0.5201,  0.5504],
        [-0.0078,  0.0139,  0.0147],
        [-0.2948,  0.5220,  0.5524],
        [-0.2709,  0.4796,  0.5075]])
```

(*): *This is referred to as [one-hot encoding](https://en.wikipedia.org/wiki/One-hot). If we have e.g. 3 possible classes: `yellow`, `green` and `blue`, we could assign the following label mapping: `yellow: 0`, `green: 1` and `blue: 2`. Then we could encode these labels as:*
$$
\text{yellow} = \begin{bmatrix}1\\0\\0\end{bmatrix}, \text{green} = \begin{bmatrix}0\\1\\0\end{bmatrix}, \text{blue} = \begin{bmatrix}0\\0\\1\end{bmatrix},
$$
*But why would we choose to do this instead of just using $0, 1, 2$ ? The reason is simple, using ordinal categorical label injects assumptions into the network that we want to avoid. The network might assume that `yellow: 0` is more different from `blue: 2` than `green: 1` because the difference in the labels is greater. We want our neural networks to output* **probability distributions over classes** *meaning that the output of the network might look something like:*
$$
\text{NN}(x) = \begin{bmatrix}0.32\\0.03\\0.65\end{bmatrix}
$$
*From this we can directly make a prediction, `0.65` is highest so the model is most confident in that the input feature corresponds to the `blue` label*


## Section 2 - Training the Network
We are now ready to train the network. Training consists of:
1. forward propagating an input feature through the network
2. Calculate the error between the prediction the network made and the actual target
3. Back-propagating the error through the network to adjust the weights.


### Section 2.1
Write a function called `W1tr, W2tr, E_total, misclassification_rate, guesses = train_nn(X_train, t_train, M, K, W1, W2, iterations, eta)` where

Inputs:
* `X_train` and `t_train` are the training data and the target values
* `M, K, W1, W2` are defined as above
* `iterations` is the number of iterations the training should take, i.e. how often we should update the weights
* `eta` is the learning rate.

Outputs:
* `W1tr`, `W2tr` are the updated weight matrices
* `E_total` is an array that contains the error after each iteration.
* `misclassification_rate` is an array that contains the misclassification rate after each iteration
* `guesses` is the result from the last iteration, i.e. what the network is guessing for the input dataset `X_train`.

The inner working of your `train_nn` function should follow this order of actions:

1. Initialize necessary variables
2. Run a loop for `iterations` iterations.
3. In each iteration we will collect the gradient error matrices for each data point. Start by initializing `dE1_total` and `dE2_total` as zero matrices with the same shape as `W1` and `W2` respectively.
4. Run a loop over all the data points in `X_train`. In each iteration we call backprop to get the gradient error matrices and the output values.
5. Once we have collected the error gradient matrices for all the data points, we adjust the weights in `W1` and `W2`, using `W1 = W1 - eta * dE1_total / N` where `N` is the number of data points in `X_train` (and similarly for `W2`).
6. For the error estimation we'll use the cross-entropy error function, *(Eq 5.74 in Bishop (Eq. 4.90 in old Bishop))*.
7. When the outer loop finishes, we return from the function

Example inputs and outputs:

*Call the function*:
```
# initialize the random seed to get predictable results
torch.manual_seed(1234)

K = 3  # number of classes
M = 6
D = train_features.shape[1]

# Initialize two random weight matrices
W1 = 2 * torch.rand(D + 1, M) - 1
W2 = 2 * torch.rand(M + 1, K) - 1
W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
    train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)
```

*Output*
* `W1tr`:
```
tensor([[-0.9578, -0.5036, -0.5738, -0.0352, -0.6735,  0.4771],
        [-0.9725, -1.1882,  0.2444,  0.4644,  0.8528,  0.0733],
        [ 0.4894, -0.3533, -1.0340,  1.3026,  0.6854,  0.8599],
        [-0.5298,  1.4409,  1.0824, -1.2925, -2.2256, -0.9583],
        [ 0.9365,  1.3075,  0.3239, -0.6886, -0.3619, -1.2948]])
```
* `W2tr`:
```
tensor([[-0.8942, -0.5361,  0.5464],
        [-0.3320,  0.0241,  0.3175],
        [-0.8556, -1.1253,  2.3209],
        [-2.7689,  0.4541, -0.4030],
        [ 1.4594, -0.0778, -2.3076],
        [ 1.9653, -1.6399, -2.4486],
        [ 2.0028, -0.5204, -0.9853]])
```
* `Etotal`:
```
tensor([2.5308, 2.3146, 2.1749, 2.0719, 1.9919, 1.9305, 1.8853, 1.8525, 1.8280,
        1.8088, 1.7928, 1.7790, 1.7667, 1.7554, 1.7448, 1.7349, 1.7253, 1.7162,
        ...
        0.5442, 0.5433, 0.5423, 0.5414, 0.5404, 0.5395, 0.5385, 0.5375, 0.5366,
        0.5356, 0.5347, 0.5337, 0.5328, 0.5318])
```
* `misclassification_rate`:
```
tensor([0.5500, 0.5500, 0.5500, 0.5500, 0.5500, 0.5500, 0.5500, 0.5500, 0.5500,
        0.5500, 0.5500, 0.5500, 0.5500, 0.5500, 0.4500, 0.4000, 0.3500, 0.3000,
        0.3000, 0.3000, 0.3000, 0.3000, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500,
        0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2500, 0.2000, 0.2000, 0.2000,
        ...
        0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500,
        0.0500, 0.0500, 0.0500, 0.0500, 0.0500])
```
* `last_guesses`:
```
tensor([0., 0., 2., 2., 0., 2., 2., 0., 1., 0., 1., 1., 2., 0., 2., 2., 2., 0., 0., 0.])
```

### Section 2.2
Write a function `guesses = test_nn(X_test, M, K, W1, W2)` where

* `X_test`: the dataset that we want to test.
* `M`: size of the hidden layer.
* `K`: size of the output layer.
* `W1, W2`: fully trained weight matrices, i.e. the results from using the `train_nn` function.
* `guesses`: the classification for all the data points in the test set `Xtest`. This should be a $(1\times N)$ vector where $N$ is the number of data points in `X_test`.

The function should run through all the data points in `X_test` and use the `ffnn` function to guess the classification of each point.

### Section 2.3

Now train your network and test it on the Iris dataset. Use 80% (as usual) for training.
At the beginning of your solution, before you load the data, use a seed that is unique to you: `torch.manual_seed(<your-seed>)`.
This will ensure your results are unique. For reproducibility, make a note of your `seed` and submit it with your results.

1. Calculate the accuracy
2. Produce a confusion matrix for your test features and test predictions
3. Plot the `E_total` as a function of iterations from your `train_nn` function.
4. Plot the `misclassification_rate` as a function of iterations from your `train_nn` function.

Submit this and **relevant discussion** in a single PDF document `2_3.pdf`.


## What to turn in to Gradescope
*Read this carefully before you submit your solution.*

You should edit `template.py` to include your own code.
 
This is an individual project, you can of course help each other out but your code should be your own.

You can use built-in python modules as you wish, 
however you are not allowed to install and import packages that are not already imported.

Files to turn in:

- `template.py`: This is your code
- `2_3.pdf`

Make sure the file names are exact. 
Submission that do not pass the first two tests in Gradescope will not be graded.


### Independent section (optional)
In this assignment we have created a naive neural network. This network could be changed in small ways that might have meaningful impact on performance. To name a few:
1. The size of the hidden layer
2. The learning rate
3. The initial values for the weights

Change the network in some way and compare the accuracy of your network for different configurations (i.e. if you decide to change the size of the hidden layer, a graph of test accuracy as a function of layer size would be ideal).

Use the PDF document to present your solution to the independent section.
