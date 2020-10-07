# -*- coding: utf-8 -*-
"""doc string"""

import csv
import numpy as np


def load_txt_data(file_name):
    with open(file_name) as f:
        rows = csv.reader(f, delimiter=',')
        data = np.array(list(rows))
    return data.astype(float)


def normalize(v):
    return (v - v.mean()) / v.std()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import linear_regression as lr

    data_path = r'test_data\city_population_profit.txt'
    test_data = load_txt_data(data_path)
    x, y = test_data[:, 0], test_data[:, 1]
    fig = plt.figure(figsize=(8, 7))
    animation = lr.display.make_animation(
        figure=fig,
        x_vector=x,
        y_vector=y,
        x_label='Population of City in 10,000s',
        y_label='Profit in $10,000s',
        learning_rate=2.4E-2,
        iteration_multiplier=10,
        max_iteration=1E3,
        starting_point=None,
        reset=True,
        reset_pause=30,
        interval=10,
        contour_center='middle',
    )
    # animation.save('test.gif', writer='pillow')
    plt.show()
