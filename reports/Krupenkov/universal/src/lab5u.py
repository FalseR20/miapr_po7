import time

import numpy as np
from uninn import Layer, NeuralNetwork


def create_row_with_switched_bit(row: np.ndarray, j: int) -> np.ndarray:
    row_copy = row.copy()
    row_copy[j] ^= 1
    return row_copy


def main():
    nn = NeuralNetwork([Layer((20, 4)), Layer((4, 3))])

    learn_x = np.array(
        [
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
        ]
    )
    learn_e = np.eye(3)
    times = 20
    print(f"- Learning {times} times -")

    start_time = time.time()

    for t in range(times):
        error = nn.learn(learn_x, learn_e).sum()
        print(f"{t + 1 : 5d}/{times} error: {error : .5e}")
        if error < 1e-20:
            print("Learning threshold crossed, breaking...")
            break
    print(f"- Learning time: {time.time() - start_time} seconds -")

    print("\nГлубокая проверка с заменой каждого бита")
    correct_amount = 0
    i: int
    row: np.ndarray
    for i, row in enumerate(learn_x):
        result = nn.go(row)
        if result.argmax() == i:
            correct_amount += 1
        probability = (result[i] / result.clip(0).sum()) * 100.0
        print(f"[{i}]: {probability:6.2f}% | ", end="")

        for j in range(20):
            result = nn.go(create_row_with_switched_bit(row, j))
            if result.argmax() == i:
                correct_amount += 1
            probability = (result[i] / result.clip(0).sum()) * 100.0
            if probability > 100.0:
                check = result
            print(f"{probability:6.2f}%", end=" ")
        print()
    print(f"Правильно {correct_amount} / 63: {correct_amount / 0.63 : .1f}%")


if __name__ == "__main__":
    main()
