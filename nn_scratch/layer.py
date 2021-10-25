import numpy as np
from . import activations

class Layer(object):
    def __init__(self,ninput,noutput,activation):
        self.ninput=ninput
        self.noutput=noutput
        self.func=activation
        self.W=np.random.random((noutput,ninput))
        self.b=np.random.random(noutput)

    def feed_forward(x_vec):
        assert(x_vec.shape[0]==self.ninput)
        self.x_vec=x_vec
        self.a_vec=self.func.call(self.W.dot(x_vec)+self.b)
        return self.a_vec

    # Back propagation using gradient descent
    def back_propagate(dC_da,lr):
        dC_dz=dC_da*self.func.derivative(self.x_vec,self.a_vec)
        dC_dW=np.full(self.noutputs,dC_da*self.a_vec)
        self.W-=lr*dC_dW
        dC_db=dC_dz*1
        self.dC_db-=lr*dC_db
        dC_da0=np.matmul(dC_da,self.sum(self.W,axes=0))
        return dC_da0
