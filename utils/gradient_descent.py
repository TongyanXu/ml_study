# -*- coding: utf-8 -*-
"""gradient descent"""

import numpy as _np
from utils.result_base import ResultBase as _ResultBase

_learning_rate = 1E-3
_max_iteration = 1E5
_precision = 1E-12
_iteration_multiplier = 1


class _GDResult(_ResultBase):
    __name__ = 'GD result'
    __slots__ = ['theta', 'loss', 'iteration', 'precision', 'converge']
    __display__ = ['converge']


class _GDStepResult(_ResultBase):
    __name__ = 'GD step result'
    __slots__ = ['theta', 'loss', 'iteration', 'theta_pre', 'loss_pre']
    __display__ = ['iteration']


def gradient_descent(loss_function, gradient_function, starting_point,
                     learning_rate=_learning_rate, max_iteration=_max_iteration,
                     precision=_precision):
    theta = starting_point.copy()
    loss_pre = loss_function(theta)
    for i in range(int(max_iteration)):
        derivatives = gradient_function(theta)
        theta -= learning_rate * derivatives
        loss = loss_function(theta)
        if loss_pre - loss < precision:
            return _GDResult(theta, loss, i + 1, precision, True)
        loss_pre = loss
    return _GDResult(theta, loss_pre, max_iteration, precision, False)


def gradient_descent_step(loss_function, gradient_function, starting_point,
                          learning_rate=_learning_rate, max_iteration=_max_iteration,
                          iteration_multiplier=_iteration_multiplier):
    i = 0
    theta_pre = starting_point.copy()
    loss_pre = loss_function(theta_pre)
    yield _GDStepResult(theta_pre, loss_pre, i, theta_pre, loss_pre)
    theta = theta_pre.copy()
    while i < max_iteration:
        for j in range(iteration_multiplier):
            gradient = gradient_function(theta)
            theta -= learning_rate * gradient
        loss = loss_function(theta)
        i += iteration_multiplier
        yield _GDStepResult(theta, loss, i, theta_pre, loss_pre)
        theta_pre, loss_pre = theta.copy(), loss.copy()


if __name__ == '__main__':

    def test_loss_function(theta):
        x, y = theta[0], theta[1]
        return (x + y - 3) ** 2 + (x + 2 * y - 5) ** 2 + 2

    def test_gradient_function(theta):
        x, y = theta[0], theta[1]
        return _np.array([2 * (x + y - 3) + 2 * (x + 2 * y - 5),
                          2 * (x + y - 3) + 4 * (x + 2 * y - 5)])

    result = gradient_descent(
        loss_function=test_loss_function,
        gradient_function=test_gradient_function,
        starting_point=_np.zeros(2),
        learning_rate=1E-3,
        precision=1E-12,
        max_iteration=1E5,
    )
    print(result)
