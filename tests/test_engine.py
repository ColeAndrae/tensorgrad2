import numpy as np
import pytest

from tensorgrad import Tensor, gradcheck, unbroadcast


SHAPE_CASES = [
    ((3, 2), (2,)),
    ((3, 2), (3, 1)),
    ((3, 2), (1, 2)),
    ((3, 2), (3, 2)),
    ((2, 3, 4), (4,)),
    ((2, 3, 4), (3, 1)),
    ((2, 3), (1, 1)),
]


def reference_unbroadcast(grad, shape):
    """Independent answer: undo numpy broadcasting by summing over the copies."""
    lead = grad.ndim - len(shape)
    grad = grad.sum(axis=tuple(range(lead)))
    axes = tuple(i for i, d in enumerate(shape) if d == 1 and grad.shape[i] != 1)
    return grad.sum(axis=axes, keepdims=True)


@pytest.mark.parametrize("grad_shape, target_shape", SHAPE_CASES)
def test_unbroadcast_shape_and_values(grad_shape, target_shape):
    grad = np.random.randn(*grad_shape)
    out = unbroadcast(grad, np.zeros(target_shape))
    assert out.shape == target_shape
    assert np.allclose(out, reference_unbroadcast(grad, target_shape))


ADD_CASES = [
    ((3, 2), (3, 2)),
    ((3, 2), (2,)),
    ((3, 2), (3, 1)),
    ((2,), (3, 2)),
    ((1, 2, 4), (1, 1, 1, 4)),
    ((2, 3, 4), (3, 1)),
]


@pytest.mark.parametrize("a_shape, b_shape", ADD_CASES)
def test_add_gradcheck(a_shape, b_shape):
    a = Tensor(np.random.randn(*a_shape))
    b = Tensor(np.random.randn(*b_shape))
    gradcheck(lambda x, y: x + y, [a, b])


def test_gradcheck_catches_a_wrong_backward():
    """The checker itself must fail on a deliberately wrong gradient."""

    def double_counting_add(x, y):
        out = x + y
        correct = out._backward

        def wrong():
            correct()
            x.grad += unbroadcast(out.grad, x.grad)

        out._backward = wrong
        return out

    a = Tensor(np.random.randn(3, 2))
    b = Tensor(np.random.randn(3, 2))
    with pytest.raises(AssertionError):
        gradcheck(double_counting_add, [a, b])
