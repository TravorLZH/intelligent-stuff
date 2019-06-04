# MNIST Classifier

My "Hello world" project of Neural Networks. It is currently based on the Keras
API, but I am going implement one **literally from scratch** after I mastered
the section of back propagation

## Prerequisites

Before trying the programs, make sure `numpy`, `opencv-python` are properly
installed.

To run `derivative.py`, you also need `matplotlib` since it is using `pyplot`
to graph functions.

## Currently Implemented

* mnist\_interpreter.py: This is a script that helps you view the labeled images
from either of the training or testing data sets. It is well-documented, so
developers are easy to understand the format of IDX file and can implement new
interpreters by themselves.

![MNIST interpreter](interpreter.png)

* derivative.py: This script shows the relationship between a function and its
derivative by using tangent line. This is a fundamental concept for my currently
incomplete **Gradient Descent** algorithm.

![Derivative plot](derivative_plot.png)

* mnist\_test.py: This is the simplest fully-connected network implemented to
classify the MNIST handwriting digits. Due to my limitation of mathematics, the
program is written using Keras API.

![MNIST classifier](mnist_test_screenshot.png)
