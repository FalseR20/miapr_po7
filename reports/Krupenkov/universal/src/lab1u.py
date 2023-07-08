import numpy as np
from creation_utils import predict_set
from uninn import *
from uninn import Layer


# Функция по условию (лаб.1 вар.9)
def function_lab1_9(x: float) -> float:
    return np.sin(8 * x) + 0.3


def main():
    layer = Layer(lens=(5, 1))  # Создание слоя
    nn = NeuralNetwork(layer)  # Создание нейронной сети с этим слоем
    learn_x, learn_e = predict_set(0, 5, 30, 0.1, function=function_lab1_9)  # Набор для предсказания функции

    for t in range(10):  # Прогон набора _ раз
        square_error = nn.learn(learn_x, learn_e, 0.08)  # Метод для обучения
        print(f"Average square error {t : 3}: {square_error}")  # Вывод на каждой итерации средней ошибки

    test_x, test_e = predict_set(3, 5, 15, 0.1, function=function_lab1_9)  # Набор для тестирования
    nn.prediction_results_table(test_x, test_e)  # Метод для красивого вывода


if __name__ == "__main__":
    main()
