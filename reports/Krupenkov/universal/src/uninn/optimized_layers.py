from typing import Optional

import numpy as np
from numpy._typing import NDArray
from uninn import Layer
from uninn.activations import LinearActivation, SigmoidActivation


class LayerSigmoid(Layer):
    """Слой нейросети с сигмоидной функцией активации"""

    def __init__(self, lengths: tuple[int, int]):
        """Слой нейросети с сигмоидной функцией активации"""
        super().__init__(lengths, activation=SigmoidActivation())

    def back_propagation(self, error: NDArray, alpha: Optional[float]) -> Optional[NDArray[float]]:
        """Обратное распространение ошибки с изменением весов, порога"""
        error_later = np.dot(error * self.y * (1 - self.y), self.w.transpose()) if not self.is_first else None

        if not alpha:
            alpha = self.adaptive_alpha(error)

        gamma = alpha * error * self.y * (1 - self.y)
        self.w -= np.dot(self.x.reshape(-1, 1), gamma.reshape(1, -1))
        self.t += gamma

        return error_later

    def adaptive_alpha(self, error) -> float:
        alpha = (
            4
            * (error**2 * self.y * (1 - self.y)).sum()
            / (1 + (self.x**2).sum())
            / np.square(error * self.y * (1 - self.y)).sum()
        )
        return alpha


class LayerLinear(Layer):
    """Слой нейросети с линейной функцией активации"""

    def __init__(self, lengths: tuple[int, int]):
        """Слой нейросети с линейной функцией активации"""
        super().__init__(lengths, activation=LinearActivation())

    def back_propagation(self, error: NDArray, alpha: Optional[float]) -> Optional[NDArray]:
        """Обратное распространение ошибки с изменением весов, порога"""
        error_later = np.dot(error, self.w.transpose()) if not self.is_first else None

        if not alpha:
            alpha = self.adaptive_alpha(error)

        gamma = alpha * error
        self.w -= np.dot(self.x.reshape(-1, 1), gamma.reshape(1, -1))
        self.t += gamma

        return error_later

    def adaptive_alpha(self, error) -> float:
        alpha = 1 / (1 + np.square(self.x).sum())
        # alpha = np.square(error).sum() / (1 + np.square(self.x).sum()) / error.sum()
        return alpha
