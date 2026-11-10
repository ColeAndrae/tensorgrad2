import numpy as np

from tensorgrad.engine import Tensor


def gradcheck(f, inputs):
    inputs = [Tensor(input.data.astype(np.float64)) for input in inputs]
    out = f(*inputs)
    R = np.random.randn(*out.shape)
    out.backward(R)

    EPS = 1e-6
    for i in range(len(inputs)):
        inputs_pos = inputs.copy()
        inputs_neg = inputs.copy()
        for idx in np.ndindex(inputs[i].shape):
            input_pos = inputs[i].data.copy()
            input_neg = inputs[i].data.copy()
            input_pos[idx] += EPS
            input_neg[idx] -= EPS
            inputs_pos[i] = Tensor(input_pos)
            inputs_neg[i] = Tensor(input_neg)
            l_pos = np.sum(f(*inputs_pos).data * R)
            l_neg = np.sum(f(*inputs_neg).data * R)
            analytical = inputs[i].grad[idx]
            numerical = (l_pos - l_neg) / (EPS * 2)
            assert np.allclose(analytical, numerical, rtol=1e-5, atol=1e-6)
