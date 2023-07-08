import numpy as np
from numpy.typing import NDArray
from uninn.activations.activation import Activation


class LinearActivation(Activation):
    def f(self, value: NDArray[float]) -> NDArray[float]:
        return value

    def fd(self, value: NDArray[float]) -> NDArray[float]:
        return np.ones(value.shape)
