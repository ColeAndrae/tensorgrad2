import numpy as np


class Tensor:

    def __init__(self, data, children=()):
        self.data = data
        self.children = set(children)
        self.grad = np.zeros(data.shape)
        self._backward = lambda : None
        self.shape = data.shape
        self.size = data.size

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += unbroadcast(out.grad, self.grad)
            other.grad += unbroadcast(out.grad, other.grad)
        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad += unbroadcast(out.grad * other.data, self.grad)
            other.grad += unbroadcast(out.grad * self.data, other.grad)
        out._backward = _backward

        return out

    def __neg__(self):
        out = Tensor(-self.data, (self,))
            
        def _backward():
            self.grad -= out.grad
        out._backward = _backward

        return out

    def __sub__(self, other):
        return self + -other

    def __pow__(self, other):
        out = Tensor(self.data ** other, (self,))

        def _backward():
            self_grad = other * self.data ** (other - 1) * out.grad
            self.grad += unbroadcast(self_grad, self.grad)
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * (other ** -1)

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            self.grad += unbroadcast(out.grad @ other.data.T, self.grad)
            other.grad += unbroadcast(self.data.T @ out.grad, other.grad)
        out._backward = _backward

        return out

    def sum(self, dim=None):
        out = Tensor(self.data.sum(dim), (self,))

        def _backward():
            self.grad += out.grad if dim is None else np.expand_dims(out.grad, dim) 
        out._backward = _backward
 
        return out

    def mean(self, dim=None):
        out = Tensor(self.data.mean(dim), (self,))

        def _backward():
            if dim is None:
                self.grad += out.grad / self.grad.size
            else:
                self.grad += np.expand_dims(out.grad, dim) / self.shape[dim] 
        out._backward = _backward

        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,))

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,))
        
        def _backward():
            self.grad += 1 / self.data * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,))

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def softmax(self, dim=-1):
        normal_sum = np.expand_dims(np.exp(self.data).sum(dim), dim)
        out = Tensor(np.exp(self.data) / normal_sum, (self,))

        def _backward():
            self.grad += out.data * (out.grad - np.sum(out.data * out.grad, axis=dim, keepdims=True))
        out._backward = _backward

        return out

    def cross_entropy(self, y):
        m = self.data.max(-1, keepdims=True)
        lse = m + np.log(np.sum(np.exp(self.data - m), -1, keepdims=True))
        out = Tensor((lse - np.sum(self.data * y, -1, keepdims=True)).mean(), (self,))

        def _backward():
            self.grad += (np.exp(self.data - lse) - y) / self.shape[0] * out.grad
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

        self.grad = np.ones(self.shape) if R is None else R
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
