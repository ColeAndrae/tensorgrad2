import numpy as np


class Tensor:

    def __init__(self, data, children=()):
        self.data = data
        self.children = set(children)
        self.grad = np.zeros(data.shape)
        self._backward = lambda : None
        self.shape = data.shape

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += unbroadcast(out.grad, self.grad)
            other.grad += unbroadcast(out.grad, other.grad)
        out._backward = _backward

        return out

    def backward(self, R=None):
        topo = []
        visited = set()
        
        def topo_sort(node):
            if node in visited:
                return
            visited.add(node)
            for child in node.children:
                topo_sort(child)
            topo.append(node)

        self.grad = self.grad if R is None else R
        topo_sort(self)
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
            return f'{self.data}'


def unbroadcast(input, target):
    target_shape = target.shape

    while len(target.shape) < len(input.shape):
        target = np.expand_dims(target, axis=0)
    for axis, dim in enumerate(target.shape):
        if dim == 1:
            input = np.sum(input, axis=axis, keepdims=True)

    return input.reshape(target_shape)
