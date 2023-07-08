from numpy.typing import NDArray

from uninn.activations.activation import Activation


class ReLuActivation(Activation):
    def f(self, value: NDArray[float]) -> NDArray[float]:
        return value.clip(0.0)

    def fd(self, value: NDArray[float]) -> NDArray[float]:
        return 1.0 * (value >= 0.0)


class ParametericReLuActivation(Activation):
    def __init__(self, a: float = 0.01):
        self.a = a

    def f(self, value: NDArray[float]) -> NDArray[float]:
        return value * (value >= 0) + self.a * value * (value < 0)

    def fd(self, value: NDArray[float]) -> NDArray[float]:
        return self.a * (value >= 0.0)
