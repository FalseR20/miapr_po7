import time

import numpy as np
from creation_utils import predict_set
from uninn import *
from uninn import Layer
from uninn.activation_functions import SigmoidActFun


def function_lab3_9(x):
    return 0.1 * np.cos(0.3 * x) + 0.08 * np.sin(0.3 * x)


def main():
    nn = NeuralNetwork(Layer(lens=(10, 4), act_fun=SigmoidActFun()), Layer(lens=(4, 1)))

    learn_x, learn_e = predict_set(0, 10, 30, 0.1, function=function_lab3_9)
    test_x, test_e = predict_set(3, 10, 15, 0.1, function=function_lab3_9)
    alpha = 0.05
    times = 30
    sep = 1000
    print(f"- Learning {times * sep} times -")

    start_time = time.time()

    for thousand in range(times):
        for _ in range(sep - 1):
            nn.learn(learn_x, learn_e, alpha)
        error = nn.learn(learn_x, learn_e, alpha)[0]
        print(f"{thousand + 1 : 5d}x{sep} error: {error : .5e}")

    print(f"- Learning time: {time.time() - start_time} seconds -")

    nn.prediction_results_table(test_x, test_e)


if __name__ == "__main__":
    main()
