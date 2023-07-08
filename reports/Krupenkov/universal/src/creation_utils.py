import os
import pickle
from typing import Callable, Optional

import numpy as np
from numpy._typing import NDArray
from uninn import NeuralNetwork


def save(nn: NeuralNetwork, filename=None) -> None:
    ans = input("Желаете сохранить? (y/n): ")
    if ans and (ans[0] == "y" or ans[0] == "н"):
        if filename is None:
            filename = input("Имя файла (*.nn): ") + ".nn"
        filename = "nn_files/" + filename
        if not os.path.exists("nn_files"):
            os.mkdir("nn_files")
        with open(filename, "wb") as file:
            pickle.dump(nn, file)
        print("Сохранено в", filename)
    else:
        print("Сохранение отклонено")


def load(filename=None) -> NeuralNetwork:
    if filename is None:
        filename = input("Имя файла (*.nn): ") + ".nn"
    filename = "nn_files/" + filename
    if os.path.exists(filename):
        with open(filename, "rb") as file:
            new_nn: NeuralNetwork = pickle.load(file)
        return new_nn
    else:
        raise FileNotFoundError


def predict_set(
    begin: float, lenght: float, count: int, step: float, function: Optional[Callable] = None
) -> tuple[NDArray[NDArray], NDArray[NDArray]]:
    """Набор обучающей выборки типа:

    x = [ [1 2 3] [2 3 4] [3 4 5] ]\n
    e = [ [4] [5] [6] ]"""

    base_array = np.arange(count + lenght)
    x = np.zeros(shape=(count, lenght))
    for i in range(count):
        x[i] = base_array[i : lenght + i]
    e = base_array[lenght : lenght + count].reshape(-1, 1)

    x = x * step + begin
    e = e * step + begin
    if function is None:
        return x, e
    else:
        return function(x), function(e)


def shuffle_set(x: NDArray[NDArray], e: NDArray[NDArray]) -> tuple[NDArray[NDArray], NDArray[NDArray]]:
    """Перемешивание набора обучающей выборки"""
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    e = e[randomize]
    return x, e
