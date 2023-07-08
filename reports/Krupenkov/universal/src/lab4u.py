import time

from creation_utils import predict_set
from lab3u import function_lab3_9
from uninn import Layer, NeuralNetwork
from uninn.activations import SigmoidActivation


def main():
    nn = NeuralNetwork([Layer(lengths=(10, 4), activation=SigmoidActivation()), Layer(lengths=(4, 1))])

    learn_x, learn_e = predict_set(0, 10, 30, 0.1, function=function_lab3_9)
    test_x, test_e = predict_set(3, 10, 15, 0.1, function=function_lab3_9)
    times = 50
    sep = 1000
    print(f"- Learning {times * sep} times -")

    start_time = time.time()

    for thousand in range(times):
        for _ in range(sep - 1):
            nn.learn(learn_x, learn_e)
        error = nn.learn(learn_x, learn_e)[0]
        print(f"{thousand + 1 : 5d}x{sep} error: {error : .5e}")
        if error < 1e-20:
            print("Learning threshold crossed, breaking...")
            break

    print(f"- Learning time: {time.time() - start_time} seconds -")

    nn.prediction_results_table(test_x, test_e)


if __name__ == "__main__":
    main()
