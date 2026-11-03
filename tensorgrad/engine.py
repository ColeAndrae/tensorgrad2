import numpy as np


class Tensor:

    def __init__(self, data, children=()):
        self.data = data
        self.children = set(children)
        self.grad = np.zeros(data.shape)
        self._backward = lambda : None

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += np.sum(out.grad, axis=0)
        out._backward = _backward

        return out
