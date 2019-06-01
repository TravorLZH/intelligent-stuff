#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Plot the tangent line at x=a
a=25

# The graph shows the function in 0<=x<=50 and -1000<=y<=1000
plt.xlim((0,50))
plt.ylim((-1000,1000))

x=np.linspace(0,50,10000)
# From a-3 to a+3 to make sure the tangent line is at least 6-unit long
x2=np.linspace(a-3,a+3,10000)

# The function itself
def f(x):
    return np.sin(1/3*x)*(-x**2+2)

# Its derivative function
def f_prime(x):
    return (-2*np.sin(1/3*x)*x+1/3*np.cos(1/3*x)*(-x**2+2))

# The function of tangent line at x=a
def tangent_line(x):
    return f_prime(a)*(x-a)+f(a)

plt.plot(x,f(x),"r-",linewidth=1,label="f(x)")
plt.plot(x2,tangent_line(x2),"b-",linewidth=1.2,
        label="tangent line of f(x) at x=%d" % a)

plt.legend()
plt.show()
