# -*- coding: utf-8 -*-
"""linear regression"""

from linear_regression import lr_by_gd as _lr_by_gd, lr_by_ne as _lr_by_ne, display


def linear_regression(x, y, mode='gradient descent', mode_cfg=None):
    if mode == 'gradient descent':
        func = _lr_by_gd.linear_regression_by_gd
    elif mode == 'normal equation':
        func = _lr_by_ne.linear_regression_by_ne
    else:
        raise ValueError('invalid linear regression mode')
    return func(x, y, mode_cfg)


class LRModel:

    def __init__(self, x, y, mode='gradient descent', mode_cfg=None):
        self._theta = linear_regression(x, y, mode, mode_cfg)

    @property
    def theta(self):
        return self._theta

    def predict(self, x):
        return x @ self._theta.T


if __name__ == '__main__':
    import numpy as _np

    test_x = _np.array([
        [1, 0],
        [1, 1],
        [1, 2],
    ])
    test_y = _np.array([
        1,
        3,
        5,
    ])
    solve = linear_regression(test_x, test_y)
    display = []
    for i in range(test_x.shape[1]):
        display.append(f'{round(solve[i], 2)} * x{i}')
    msg = ' + '.join(display) + ' = y'
    print(msg)

