#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt

w=np.array([1,1])
b=-0.2
xs=np.array([
    [1,1],
    [0,0],
    [1,0],
    [0,1]
])

# OR Gate
ys=np.array([
    1,
    0,
    1,
    1
])

def process(x):
    return activate(x.dot(w)+b)

def activate(z):
    a= z>0
    return a.astype(np.int8)

def line(x0):
    return -(b+w[0]*x0)/w[1]

yps=process(xs)

for i,x in enumerate(xs):
    print("x={}, y={}, yp={}".format(x.tolist(),ys[i],yps[i]))

plt.figure("Perceptron")
plt.title("Perceptron")
plt.xlim(-0.2,5)
plt.ylim(-0.2,5)

x0s=np.linspace(-0.5,1.5,20)
plt.plot(x0s,line(x0s),"r-")

for i,pt in enumerate(xs):
    style="go" if ys[i]==1 else "ko"
    plt.plot(pt[0],pt[1],style)

plt.show()
