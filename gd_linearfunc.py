#!/usr/bin/env python3
"""
GRADIENT DESCENT ALGORITHM

Gradient Descent Algorithm is a method to find the minimum value of a function,
and it is widely adopted in machine learning, especially deep learning. To make
an algorithm actually "learn" after processing data, a concept of minimizing
cost, or loss, is introduced. After a training procedure, the algorithm outputs a value or an array of values. Then it compares the predicted values to the
actual values, and the difference between each predicted and correct value is
called "cost" (Some call it "loss", but we will use "cost" in the text below).

Since we are doing a lot of trainings, the cost becomes a function, and it
reaches its minimum if predicted values and correct values are the same. Some
believe that modifying the predicted values to the correct values can achieve
this goal, but this eventually leads to "memorizing" the answers. Instead we
modify the parameters of the algorithm (e.g. weights and biases in deep learningalgorithms) by using gradient descent. It is like let a ball roll in the curve
of a function.

In this program, we are going to use gradient descent to fit a linear function
(y=kx+b where k!=0).
"""
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

# y=3x+2 is used to create correct y values with respect to x values
def f(x):
    return 3*x+2

k=np.random.random()*20.0 - 10.0    # Generate random numbers between [-10,10]
b=np.random.random()*20.0 - 10.0

def f_learn(x):
    return k*x+b

def calculate_cost(y,yp):
    return (yp-y)**2

def cost_prime(y,yp):
    return 2*(yp-y)

def gradient_descent(dc_dk,dc_db,step=0.0001):
    global k,b
    k-=dc_dk*step
    b-=dc_db*step

# The training set
x_train=np.linspace(0,100,101000)
y_train=f(x_train)+np.random.random()-0.5

# The testing set
x_test=np.linspace(-20,20,100)
y_test=f(x_test)

epochs=100  # Number of iterations
batch_size=1     # Batch size (self-explanatory)

if len(sys.argv)>=2:
    batch_size=int(sys.argv[1])

if batch_size==1:
    print("Using Stochastic GD")
elif batch_size==len(x_train):
    print("Using Batch GD")
else:
    print("Using Mini-batch GD, batch_size=%d" % batch)

if os.path.exists("./logs/")==False:
    os.mkdir("logs")

print("Before learning: k=%f, b=%f" % (k,b))

costs=[]

for epoch_no in range(epochs):
    n=0
    log_file=open("logs/epoch_"+str(epoch_no+1)+".log","w")
    total_cost=0
    gk=0.0
    gb=0.0
    try:
        for x,y in zip(x_train,y_train):
            yp=f_learn(x)
            n+=1
            total_cost+=calculate_cost(y,yp)
            cost=total_cost/n
            log_file.write("%d: k=%f, b=%f, y=%f, yp=%f, cost=%f\n"
                    % (n,k,b,y,yp,cost))
            # This part calculates gradients
            dc_dyp=cost_prime(y,yp)
            dyp_db=1
            dyp_dk=x
            dc_dk=dc_dyp * dyp_dk   # Chain Rule
            dc_db=dc_dyp * dyp_db
            gk+=dc_dk
            gb+=dc_db
            if (n % batch_size)==0:
                gradient_descent(gk/batch_size,gb/batch_size)
                gk=0.0
                gb=0.0
        log_file.close()
        avg_cost=total_cost/n
        costs.append(avg_cost)
        #print("Epoch #%d completes, k=%f, b=%f, cost=%f" %
        #        (epoch_no+1,k,b,avg_cost))
    except KeyboardInterrupt:
        print("Epoch #%d interrupted cost=%f, terminate training"
                % (epoch_no+1,total_cost/n))
        break

print("Learned k=%f, b=%f" % (k,b))

plt.subplot(121)
plt.plot(range(1,len(costs)+1),costs)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.subplot(122)
plt.plot(x_test,f_learn(x_test),"g-", \
    label="Linear Regression: $y_{learn}=%fx+%f" % (k,b))
plt.plot(x_test,f(y_test),label="To be learnt: $y=3x+2$")
plt.legend()
plt.show()

# Start testing...
y_predict=f_learn(x_test)
total_cost=0.0
for y,yp in zip(y_test,y_predict):
    total_cost+=calculate_cost(y,yp)

cost=total_cost / len(x_test)

print("Test result: cost=%f" % cost)
