import numpy as np

from tensorgrad import Tensor, gradcheck, unbroadcast


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
        A = Tensor(np.random.rand(*shape_A))
        B = Tensor(np.random.rand(*shape_B))
        gradcheck(lambda a, b: a + b, [A, B])
    for shape_A, shape_B in SHAPES:
            A = Tensor(np.random.rand(*shape_A))
            gradcheck(lambda a : a + a, [A])

def test_mul():
    for shape_A, shape_B in SHAPES:
        A = Tensor(np.random.rand(*shape_A))
        B = Tensor(np.random.rand(*shape_B))
        gradcheck(lambda a, b: a * b, [A, B])
    for shape_A, shape_B in SHAPES:
        A = Tensor(np.random.rand(*shape_A))
        gradcheck(lambda a : a * a, [A])

def test_neg():
    for shape_A, _ in SHAPES:
        A = Tensor(np.random.rand(*shape_A))
        gradcheck(lambda a : -a, [A])

        