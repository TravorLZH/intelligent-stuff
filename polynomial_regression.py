#!/usr/bin/env python3
"""
Polynomial regression with exponent of 3, in which the polynomial looks like:

    y=a+bx+cx^2+dx^3

For N of samples, we have

    y1=a+bx1+cx1^2+dx1^3
    y2=a+bx2+cx2^2+dx2^3
    ...
    yN=a+bxN+cxN^2+dxN^3

Using a simple linear algebraic representation, we have:

    y1=[1 x1 x1^2 x1^3] dot [a b c d]
    y2=[1 x2 x2^2 x3^3] dot [a b c d]
    ...
    yN=[1 xN xN^2 xN^3] dot [a b c d]

And if we use matrices, things get more clean:

      [1 x1 x1^2 x1^3]
      [1 x2 x2^2 x2^3]
    X=[..............]
      [..............]
      [1 xN xN^2 xN^3]

    p=[a b c d]

    y=Xp

    X^T
"""
import numpy as np
from matplotlib import pyplot as plt

x=np.arange(0,1,0.001)
y0=np.cos(3*x)
y=np.random.normal(loc=y0,scale=0.25,size=None)   # Up/Down 0.5

plt.figure("Polynomial Regression")
plt.title("Polynomial Regression")

# Create the t

plt.scatter(x,y,c="k")
plt.plot(x,y0,"r")

plt.show()
