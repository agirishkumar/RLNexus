import numpy as np
from rlnexus.linalg.core import Tensor

class Module:
    """
    Base class for all neural network modules.
    """
    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def parameters(self):
        return []

class Linear(Module):
    """
    A linear transformation layer.
    """
    def __init__(self, in_features, out_features, bias=True):
        self.weight = Tensor(np.random.randn(out_features, in_features) * np.sqrt(2. / in_features))
        self.bias = Tensor(np.zeros(out_features)) if bias else None

    def __call__(self, x):
        # x: (batch_size, in_features)
        # weight: (out_features, in_features)
        # bias: (out_features,)
        
        out = x.dot(self.weight.T)
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

class ReLU(Module):
    """
    ReLU activation function.
    """
    def __call__(self, x):
        return x.relu()

    def parameters(self):
        return []

class Sequential(Module):
    """
    A sequential container for modules.
    Modules will be added to it in the order they are passed in the constructor.
    """
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

class SGD:
    """
    Stochastic Gradient Descent optimizer.
    """
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = None
