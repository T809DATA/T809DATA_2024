# Applying Neural Networks
The goal of this exercise is to get to know PyTorch and its main features. Currently there are two popular deep learning frameworks: PyTorch and TensorFlow. TensorFlow is developed by Google Brain and actively used at Google both for research and in production. TensorFlow has been considered by many to be the state of the art in deep learning frameworks ever since it was released open source in 2015. PyTorch is a more recent addition to that market, originally released in 2016. However, the popularity of PyTorch has been increasing rapidly and it is now considered the [biggest competitor](https://hub.packtpub.com/dl-wars-pytorch-vs-tensorflow/) [to TensorFlow](https://towardsdatascience.com/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b)


We will use PyTorch in this exercise, mainly because of how tightly PyTorch is integrated with the Python language and because of how much easier it is to setup PyTorch and run it, in comparison with TensorFlow.

**Note**: This assignment is slightly different from the other assignments in multiple ways. There is no `template.py` to fill in since you will be following the tutorial below. 

Return a single PDF document with all the solutions to this exercise. If you go for the independent section, add those experiments and results to the PDF document as well.  If you have some python code for the independent section, submit that as `indep.py`, but that is the only code you might need to submit, we don't need your code for the tutorial itself.

*All questions should be answered in a PDF document `report.pdf`.*

## Section 1
### Section 1.1

**Before starting, take a look at `example.py` Try running that code to ensure that you have installed PyTorch and everything is working correctly.**

Your first task is to go through the initial tutorial for PyTorch to get a feel for how it works. The tutorial can be found [here](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

One of the modules in the tutorial is image classification (Training a classifier). Your task will be to classify multiple types of objects from images as inputs, for example cats, dogs, birds, airplanes and so on.

Questions
1. How well does your classification work? Plot the misclassification rate for each category onto the same plot.
2. Calculate the confusion matrix for the image classification task.
3. Explain what Autograd is and how it works?

## Section 2
### Section 2.1
The neural network used in the first example is a typical neural network with hidden layers. For text generation we need a bit more complicated networks, for example RNN.

Your second task is to use the [tutorial for text generation](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html). You might also find the [text classification tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) helpful.

Questions:
1. What is RNN?
2. Why do we use RNN when we are working with text?
3. In your opinion, how well does the text generation work? 
4. Name three other domains where RNNs are suitable model types for regression/classification


## What to turn in to Gradescope
*Read this carefully before you submit your solution.*

This is an individual project so you can of course help each other out but your report should be your own.

Files to turn in:

- `report.pdf`

Make sure the file names are exact. 


## Independent section (optional)
This is an open ended independent question. You are welcome to try other tutorials in the PyTorch documentation, design a model that achieves some task on Icelandic text, create your own model like the one in `example.py` or anything else you come up with.

Turn in your code as `indep.py` and add your experiments and results to the PDF document.