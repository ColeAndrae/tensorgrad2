import numpy as np

from tensorgrad import Tensor, gradcheck


SHAPES = [[(3, 2), (3, 2)],
          [(2,), (3, 2)],
          [(3, 2), (2,)],
          [(1,), (3, 2, 1)],
          [(3, 2, 1), (1,)],
          [(2, 1), (3, 2, 1)],
          [(3, 2, 1), (2, 1)],
          [(1, 3), (2, 1)]]


def test_add():
    for shape_A, shape_B in SHAPES:
        A = Tensor(np.random.randn(*shape_A))
        B = Tensor(np.random.randn(*shape_B))
        gradcheck(lambda a, b: a + b, [A, B])
    for shape_A, shape_B in SHAPES:
            A = Tensor(np.random.randn(*shape_A))
            gradcheck(lambda a : a + a, [A])

def test_mul():
    for shape_A, shape_B in SHAPES:
        A = Tensor(np.random.randn(*shape_A))
        B = Tensor(np.random.randn(*shape_B))
        gradcheck(lambda a, b: a * b, [A, B])
    for shape_A, shape_B in SHAPES:
        A = Tensor(np.random.randn(*shape_A))
        gradcheck(lambda a : a * a, [A])

def test_neg():
    for shape_A, _ in SHAPES:
        A = Tensor(np.random.randn(*shape_A))
        gradcheck(lambda a : -a, [A])

def test_sub():
    for shape_A, shape_B in SHAPES:
        A = Tensor(np.random.randn(*shape_A))
        B = Tensor(np.random.randn(*shape_B))
        gradcheck(lambda a, b: a - b, [A, B])
    for shape_A, shape_B in SHAPES:
        A = Tensor(np.random.randn(*shape_A))
        gradcheck(lambda a : a - a, [A])

def test_pow():
    for shape_A, _ in SHAPES:
        A = Tensor(np.random.randn(*shape_A))
        gradcheck(lambda a : a ** 3, [A])
        A = Tensor(np.random.rand(*shape_A))
        gradcheck(lambda a : a ** -1.5, [A])

def test_truediv():
    for shape_A, shape_B in SHAPES:
        A = Tensor(np.random.randn(*shape_A))
        B = Tensor(np.random.randn(*shape_B))
        gradcheck(lambda a, b: a / b, [A, B])
    for shape_A, shape_B in SHAPES:
        A = Tensor(np.random.randn(*shape_A))
        gradcheck(lambda a : a / a, [A])

        