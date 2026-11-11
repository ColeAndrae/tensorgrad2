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

MATMUL_SHAPES = [[(3, 2), (2, 3)],
                 [(2, 3), (3, 2)],
                 [(2, 1), (1, 4)],
                 [(3, 3), (3, 3)],
                 [(1, 3), (3, 9)]]

CROSS_ENTROPY_SHAPES = [(1, 10),
                        (2, 3),
                        (3, 3),
                        (9, 3),
                        (8, 10),
                        (1000, 5)]


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

def test_matmul():
    for shape_A, shape_B in MATMUL_SHAPES:
        A = Tensor(np.random.randn(*shape_A))
        B = Tensor(np.random.randn(*shape_B))
        gradcheck(lambda a, b: a @ b, [A, B])

def test_sum():
    for shape_A, _ in SHAPES:
        A = Tensor(np.random.randn(*shape_A))
        for dim in range(len(A.shape)):
            gradcheck(lambda a : a.sum(dim), [A])
        gradcheck(lambda a : a.sum(), [A])

def test_mean():
    for shape_A, _ in SHAPES:
        A = Tensor(np.random.randn(*shape_A))
        for dim in range(len(A.shape)):
            gradcheck(lambda a : a.mean(dim), [A])
        gradcheck(lambda a : a.mean(), [A])

def test_exp():
    for shape_A, _ in SHAPES:
        A = Tensor(np.random.randn(*shape_A))
        gradcheck(lambda a : a.exp(), [A])

def test_log():
    for shape_A, _ in SHAPES:
        A = Tensor(np.random.rand(*shape_A) + 0.5)
        gradcheck(lambda a : a.log(), [A])

def test_relu():
    for shape_A, _ in SHAPES:
        A = Tensor(np.random.randn(*shape_A))
        gradcheck(lambda a : a.relu(), [A])

def test_softmax():
    for shape_A, _ in SHAPES:
        A = Tensor(np.random.randn(*shape_A))
        for dim in range(len(A.shape)):
            gradcheck(lambda a : a.softmax(dim), [A])
        gradcheck(lambda a : a.softmax(), [A])

def test_cross_entropy():
    for shape_A in CROSS_ENTROPY_SHAPES:
        A = Tensor(np.random.randn(*shape_A))
        N, C = shape_A
        b = np.eye(C)[np.random.randint(C, size=N)]
        gradcheck(lambda a : a.cross_entropy(b), [A])