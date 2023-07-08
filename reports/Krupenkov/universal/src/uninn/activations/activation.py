import abc

from numpy.typing import NDArray


class Activation(abc.ABC):
    @abc.abstractmethod
    def f(self, value: NDArray[float]) -> NDArray[float]:
        """Activation function (f)"""
        pass

    @abc.abstractmethod
    def fd(self, value: NDArray[float]) -> NDArray[float]:
        """Activation function derivative (f')"""
        pass
