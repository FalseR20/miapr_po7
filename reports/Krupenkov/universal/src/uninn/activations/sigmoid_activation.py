import numpy as np
from numpy.typing import NDArray
from uninn.activations.activation import Activation


class SigmoidActivation(Activation):
    def f(self, value: NDArray[float]) -> NDArray[float]:
        return 1.0 / (1.0 + np.exp(-value))

    def fd(self, value: NDArray[float]) -> NDArray[float]:
        return value * (1.0 - value)
