from typing import Optional

import numpy as np
from numpy.typing import NDArray
from uninn.activations import Activation, LinearActivation, SigmoidActivation


class Layer:
    """Слой нейросети"""

    def __init__(
        self,
        lengths: tuple[int, int],
        activation: Activation = None,
        w: NDArray[float] = None,
        t: NDArray[float] = None,
    ):
        """
        :param lengths: количество нейронов этого и следующего слоя
        :param lengths: экземпляр класса функции активации
        :param w: веса
        :param t: пороги
        """
        self.lengths = lengths
        self.activation: Activation = activation or LinearActivation()
        self.w: NDArray[float] = w or np.random.uniform(-0.5, 0.5, self.lengths)
        self.t: NDArray[float] = t or np.random.uniform(-0.5, 0.5, self.lengths[1])
        self.x: Optional[NDArray[float]] = None
        self.s: Optional[NDArray[float]] = None
        self.y: Optional[NDArray[float]] = None
        self.is_first = False

    def go(self, x: NDArray[float]) -> NDArray[float]:
        """Прохождение слоя"""
        self.x = x
        self.s: NDArray[float] = np.dot(x, self.w) - self.t
        self.y: NDArray[float] = self.activation.f(self.s)
        return self.y

    def back_propagation(
        self, error: NDArray[float], alpha: Optional[float]
    ) -> Optional[NDArray[float]]:
        """Обратное распространение ошибки с изменением весов, порога"""
        error_later: Optional[NDArray[float]] = (
            np.dot(error * self.activation.fd(self.y), self.w.transpose()) if not self.is_first else None
        )

        if not alpha:
            alpha = self.adaptive_alpha(error)

        gamma = alpha * error * self.activation.fd(self.y)
        self.w -= np.dot(self.x.reshape(-1, 1), gamma.reshape(1, -1))
        self.t += gamma

        return error_later

    def adaptive_alpha(self, error: NDArray[float]) -> float:
        if not hasattr(self, "d_f_act_0"):
            setattr(self, "d_f_act_0", self.activation.fd(self.activation.f(np.zeros(error.shape))))
        alpha = (
            (error ** 2 * self.activation.fd(self.y)).sum()
            / getattr(self, "d_f_act_0")
            / (1 + (self.x**2).sum())
            / ((error * self.activation.fd(self.y)) ** 2).sum()
        )
        return alpha


class LayerSigmoid(Layer):
    """Слой нейросети с сигмоидной функцией активации"""

    def __init__(self, lengths: tuple[int, int]):
        """Слой нейросети с сигмоидной функцией активации"""
        super().__init__(lengths, activation=SigmoidActivation())

    def back_propagation(
        self, error: NDArray, alpha: Optional[float], is_first_layer=False
    ) -> Optional[NDArray[float]]:
        """Обратное распространение ошибки с изменением весов, порога"""
        error_later = np.dot(error * self.y * (1 - self.y), self.w.transpose()) if not is_first_layer else None

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

    def back_propagation(self, error: NDArray, alpha: Optional[float], is_first_layer=False) -> Optional[NDArray]:
        """Обратное распространение ошибки с изменением весов, порога"""
        error_later = np.dot(error, self.w.transpose()) if not is_first_layer else None

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
