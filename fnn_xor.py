#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(y):
    return y*(1.0-y)

train_xs=np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

train_ys=np.array([0,1,1,0])

input_dim=2
hidden_dim=2
output_dim=1

# Training parameters
epochs=100000
batch_size=len(train_xs)

# Gradient descent parameters
gd_step=0.01

# Network parameters
np.random.seed(0)
W1=np.random.normal(0,1,(hidden_dim,input_dim))
W2=np.random.normal(0,1,(output_dim,hidden_dim))
b1=np.random.random(hidden_dim)
b2=np.random.random(output_dim)
#print("Initial shape: W1={}, W2={}, b1={}, b2={}".format(W1.shape,W2.shape,
#    b1.shape,b2.shape))

def feed_forward(x,train=False):
    a0=x
    a1=sigmoid(np.dot(W1,a0)+b1)
    a2=sigmoid(np.dot(W2,a1)+b2)
    #print("a0: {}, a1: {}, a2: {}".format(a0.shape,a1.shape,a2.shape))
    if train: return (a0,a1,a2)
    return a2

def cross_entropy(y,yp):
    return (-y*np.log(yp))-((1-y)*np.log(1-a2))

def sigmoid_cross_entropy_prime(y,yp):
    return yp-y

def back_propagate(a0,a1,a2,y):
    global W1,W2,b1,b2
    dz2=a2-y
    dW2=dz2*a1
    dz1=np.multiply(np.dot(W2.T,dz2),sigmoid_prime(a1))
    dW1=dz1.dot(x)
    db1=dz1
    db2=dz2
    W1=W1-gd_step*dW1
    W2=W2-gd_step*dW2
    b1=b1-gd_step*db1
    b2=b2-gd_step*db2
    #print("Shapes: W1={}, W2={}, b1={}, b2={}".format(W1.shape,W2.shape,
    #    b1.shape,b2.shape))

costs=[]
n=1
try:
    for i in range(1,epochs+1):
        n=i
        total_cost=0.0
        for x,y in zip(train_xs,train_ys):
            a0,a1,a2=feed_forward(x,train=True)
            #print(a2.shape)
            total_cost+=cross_entropy(y,a2)
            back_propagate(a0,a1,a2,y)
        avg_cost=total_cost/batch_size
        costs.append(avg_cost)
except KeyboardInterrupt:
    print("Interrupted at epoch #%d" % n)

plt.figure("XOR Neural Network")
plt.title("XOR Neural Network Training Result")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.plot(range(1,len(costs)+1),costs)
plt.show()
