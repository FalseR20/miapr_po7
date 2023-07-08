from typing import Optional

import numpy as np
from numpy.typing import NDArray
from uninn import Layer


class NeuralNetwork:
    def __init__(self, layers: list[Layer]) -> None:
        """Создание нейросети с заданием массива слоев"""
        self.layers: list[Layer] = layers
        self.layers[0].is_first = True

    def go(self, x: NDArray[float]) -> NDArray[float]:
        """Прохождение всех слоев нейросети"""
        for layer in self.layers:
            x = layer.go(x)
        return x

    def learn(self, x: NDArray[float], e: NDArray[float], alpha: Optional[float] = None) -> NDArray[float]:
        """Обучение наборами обучающих выборок

        - x: (n, len_in) ... [[1 2] [2 3]]
        - e: (n, len_out) ....... [[3] [4]]
        """

        square_error = np.zeros(self.layers[-1].lengths[1])
        for i in range(len(e)):
            y: NDArray[float] = self.go(x[i])
            error: NDArray[float] = y - e[i]
            square_error += error**2 / 2
            for layer in reversed(self.layers):
                error = layer.back_propagation(error, alpha)
        return square_error / self.layers[-1].lengths[1]

    def prediction_results_table(self, x: NDArray[float], e: NDArray[float]) -> None:
        """Красивый вывод прогона тестирующей выборки"""
        print("                эталон        выходное значение                  разница         среднекв. ошибка")
        y = self.go(x)
        y = y.reshape(-1)
        e = e.reshape(-1)
        delta = y - e
        square_error = delta**2 / 2
        for i in range(len(e)):
            print(f"{e[i] : 22}{y[i] : 25}{delta[i] : 25}{square_error[i] : 25}")
        print(f"\n-- Final testing square error: {np.average(square_error) * 2} --")
