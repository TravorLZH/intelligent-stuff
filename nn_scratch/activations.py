import numpy as np

def sigmoid_func(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(y):
    return y*(1-y)

def relu_func(x):
    return np.maximum(x,0)

def relu_prime(x):
    x[x<=0]=0
    x[x>0]=1
    return x

class Activation():
    def __init__(self,func,d_func,y_deriv=False):
        self.func=func
        self.d_func=d_func
        self.y_deriv=y_deriv
    def call(x):
        return self.func(x)
    def derivative(x,y):
        # Since sigmoid(x) takes y as its derivative, we can optimize for it
        if self.y_deriv:
            return self.d_category(y)
        else:
            return self.d_category(x)

sigmoid=Activation(sigmoid_func,sigmoid_prime,y_deriv=True)
relu=Activation(relu_func,relu_prime)
