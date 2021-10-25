from layer import Layer
from activations import relu

class Network(object):
    def __init__(self,ninputs,lr):
        self.ninputs=ninputs
        self.tmp_inputs=ninputs
        self.layers=[]
        self.lr=lr

    def append(noutputs,activation):
        self.layers.append(Layer(self.tmp_inputs,noutputs,activation))
        self.tmp_inputs=noutputs

    def forward_propagate(x_vec):
        self.y_vec=x_vec
        for layer in layers:
            self.y_vec=layer.feed_forward(y_vec)
        return self.y_vec

    def back_propagate(y):
        dC_da=2*(self.y_vec-y)
        for layer in reversed(self.layers):
            dC_da=layer.back_propagate(dC_da,lr)
