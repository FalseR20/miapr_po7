from typing import Optional

import numpy as np
from numpy._typing import NDArray
from uninn.activation_functions import ActFun, LinearActFun, SigmoidActFun


class Layer:
    """Слой нейросети"""

    def __init__(
        self,
        lens: tuple[int, int],
        act_fun: ActFun = None,
        w=None,
        t=None,
    ):
        """
        - lens (количество нейронов этого и следующего слоя)
        - функции активации
        """
        self.lens = lens
        self.act_fun = act_fun or LinearActFun()
        self.w = np.random.uniform(-0.5, 0.5, lens) if w is None else w
        self.t = np.random.uniform(-0.5, 0.5, lens[1]) if t is None else t

    def go(self, x: NDArray) -> NDArray:
        """Прохождение слоя"""
        self.x = x
        self.s: NDArray = np.dot(x, self.w) - self.t
        self.y: NDArray = self.act_fun.f(self.s)
        return self.y

    def back_propagation(self, error: NDArray, alpha: Optional[float], is_first_layer=False) -> Optional[NDArray]:
        """Обратное распространение ошибки с изменением весов, порога"""
        error_later = np.dot(error * self.act_fun.d(self.y), self.w.transpose()) if not is_first_layer else None

        if not alpha:
            alpha = self.adaptive_alpha(error)

        gamma = alpha * error * self.act_fun.d(self.y)
        self.w -= np.dot(self.x.reshape(-1, 1), gamma.reshape(1, -1))
        self.t += gamma

        return error_later

    def adaptive_alpha(self, error: NDArray) -> float:
        if not hasattr(self, "d_f_act_0"):
            self.d_f_act_0 = self.act_fun.d(self.act_fun.f(0))
        alpha = (
            (error**2 * self.act_fun.d(self.y)).sum()
            / self.d_f_act_0
            / (1 + (self.x**2).sum())
            / ((error * self.act_fun.d(self.y)) ** 2).sum()
        )
        return alpha


class LayerSigmoid(Layer):
    """Слой нейросети с сигмоидной функцией активации"""

    def __init__(self, lens: tuple[int, int]):
        """Слой нейросети с сигмоидной функцией активации"""
        super().__init__(lens, act_fun=SigmoidActFun())

    def back_propagation(self, error: NDArray, alpha: Optional[float], is_first_layer=False) -> Optional[NDArray]:
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

    def __init__(self, lens: tuple[int, int]):
        """Слой нейросети с линейной функцией активации"""
        super().__init__(lens, act_fun=LinearActFun())

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
