# -*- coding: utf-8 -*-
"""optimizer - gradient descent (steepest descent)"""

import numpy as _np
import optimizer.base as _base

# default config
_learning_rate = 1E-3
_max_iteration = 1E5
_g_tolerance = 1E-5
_g_norm = 2
_iteration_multiplier = 1


class GradientDescent(_base.OptimizerBase):
    """simple gradient descent (steepest descent) without line search algorithm"""

    __slots__ = ['l_rate', 'max_iter', 'g_tol', 'g_norm']
    __default__ = {
        'l_rate': _learning_rate,
        'max_iter': _max_iteration,
        'g_tol': _g_tolerance,
        'g_norm': _g_norm
    }

    def __init__(self, l_rate: (float, int) = None, max_iter: (float, int) = None,
                 g_tol: (float, int) = None, g_norm: int = None):
        super().__init__(l_rate, max_iter, g_tol, g_norm)
        self.max_iter = int(self.max_iter)

    def optimize(self, loss: callable, init_val: _np.array,
                 jacob: callable = None, hess: callable = None
                 ) -> _base.OptimizeResult:
        theta = init_val.copy()
        g_norm = None
        for i in range(self.max_iter):
            gradient = jacob(theta)
            theta -= self.l_rate * gradient
            g_norm = _np.linalg.norm(gradient, self.g_norm)
            if g_norm < self.g_tol:
                norm_inst = _base.NormValue(self.g_norm, g_norm)
                return _base.OptimizeResult(theta, norm_inst, i + 1, True)
        norm_inst = _base.NormValue(self.g_norm, g_norm)
        return _base.OptimizeResult(theta, norm_inst, self.max_iter, False)


class GradientDescentSBS(_base.OptimizerSBS):
    """step-by-step of gradient descent (steepest descent) without line search algorithm"""

    __slots__ = ['l_rate', 'max_iter', 'iter_mul']
    __default__ = {
        'l_rate': _learning_rate,
        'max_iter': _max_iteration,
        'iter_mul': _iteration_multiplier,
    }

    def __init__(self, l_rate: (float, int) = None, max_iter: (float, int) = None,
                 iter_mul: int = None):
        super().__init__(l_rate, max_iter, iter_mul)

    def make(self, loss: callable, init_val: _np.array,
             jacob: callable = None, hess: callable = None,
             ) -> _base.OptimizeStepResult:
        i = 0
        theta_pre = init_val.copy()
        theta = theta_pre.copy()
        yield _base.OptimizeStepResult(theta, i, theta_pre)
        while i < self.max_iter:
            for j in range(self.iter_mul):
                gradient = jacob(theta)
                theta -= self.l_rate * gradient
            i += self.iter_mul
            yield _base.OptimizeStepResult(theta, i, theta_pre)
            theta_pre = theta.copy()


if __name__ == '__main__':
    def test_loss_function(theta):
        x, y = theta[0], theta[1]
        return (x + y - 3) ** 2 + (x + 2 * y - 5) ** 2 + 2


    def test_gradient_function(theta):
        x, y = theta[0], theta[1]
        return _np.array([2 * (x + y - 3) + 2 * (x + 2 * y - 5),
                          2 * (x + y - 3) + 4 * (x + 2 * y - 5)])


    optimizer = GradientDescent(
        l_rate=1E-3,
        max_iter=1E5,
        g_tol=1E-5,
        g_norm=2,
    )

    result = optimizer.optimize(
        loss=test_loss_function,
        init_val=_np.zeros(2),
        jacob=test_gradient_function,
    )
    print(result)
