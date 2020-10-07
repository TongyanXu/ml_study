# -*- coding: utf-8 -*-
"""doc string"""

import numpy as _np
import logging as _lg


def simple_gradient_descent(loss_function, gradient_function, variable_count,
                            learning_rate=1E-2, precision=1E-10, max_iter=1E5,
                            starting_point=None):
    theta_new = starting_point or _np.zeros(variable_count)
    loss_new = loss_function(theta_new)
    for i in range(int(max_iter)):
        theta_cur, loss_cur = theta_new, loss_new
        derivatives = gradient_function(theta_cur)
        theta_new = theta_cur - learning_rate * derivatives
        loss_new = loss_function(theta_new)
        if abs(loss_new - loss_cur) < precision:
            return theta_new
    return theta_new


def _simple_gradient_descent(loss_function, gradient_function, variable_count,
                             learning_rate=1E-2, precision=1E-10, max_iter=1E5,
                             starting_point=None, verbose=False):
    logger = _lg.getLogger('gradient_descent')
    if verbose:
        logger.debug('running gradient descent')
        logger.debug(f'learning rate: {learning_rate}')
        logger.debug(f'precision: {precision}')
        logger.debug(f'max iteration: {max_iter}')
    theta_cur = theta_new = starting_point or _np.zeros(variable_count)
    if verbose:
        logger.debug(f'starting point: {theta_cur.tolist()}')
    loss_cur = loss_new = loss_function(theta_cur)
    for i in range(int(max_iter)):
        theta_cur, loss_cur = theta_new, loss_new
        derivatives = gradient_function(theta_cur)
        theta_new = theta_cur - learning_rate * derivatives
        loss_new = loss_function(theta_new)
        if abs(loss_new - loss_cur) < precision:
            if verbose:
                logger.debug(f'converge after {i + 1} iteration(s)')
                logger.debug(f'minimum loss: {loss_new}')
                logger.debug(f'converge error: {loss_new - loss_cur}')
            return theta_new
    if verbose:
        logger.warning(f'failed to converge after {max_iter} iteration(s)')
        logger.warning(f'minimum loss: {loss_new}')
        logger.warning(f'converge error: {loss_new - loss_cur}')
    return theta_new


if __name__ == '__main__':
    import time
    _lg.basicConfig(level=_lg.DEBUG)

    def test_loss_function(theta):
        x, y = theta[0], theta[1]
        return (x + y - 3) ** 2 + (x + 2 * y - 5) ** 2 + 2

    def test_gradient_function(theta):
        x, y = theta[0], theta[1]
        return _np.array([2 * (x + y - 3) + 2 * (x + 2 * y - 5),
                          2 * (x + y - 3) + 4 * (x + 2 * y - 5)])

    result = _simple_gradient_descent(
        loss_function=test_loss_function,
        gradient_function=test_gradient_function,
        variable_count=2,
        learning_rate=1E-2,
        precision=1E-10,
        max_iter=1E3,
        starting_point=None,
        verbose=True,
    )
    time.sleep(0.1)
    print(result)
