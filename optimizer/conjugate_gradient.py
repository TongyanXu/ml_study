# -*- coding: utf-8 -*-
"""optimizer - conjugate descent"""

import numpy as _np
import optimizer.base as _base

_learning_rate = 1E-2
_max_iteration = 1E5
_g_tolerance = 1E-5
_g_norm = 2
_iteration_multiplier = 1


class ConjugateGradient(_base.OptimizerBase):
    __slots__ = ['l_rate', 'max_iter', 'g_tol', 'g_norm']
    __default__ = {
        'l_rate': _learning_rate,
        'max_iter': _max_iteration,
        'g_tol': _g_tolerance,
        'g_norm': _g_norm,
    }

    def __init__(self, l_rate: (float, int) = None, max_iter: (float, int) = None,
                 g_tol: (float, int) = None, g_norm: int = None):
        super().__init__(l_rate, max_iter, g_tol, g_norm)
        self.max_iter = int(self.max_iter)

    def optimize(self, loss: callable, init_val: _np.array,
                 jacob: callable = None, hess: callable = None
                 ) -> _base.OptimizeResult:
        theta = init_val.copy()
        p, g_square_pre, g_norm = 0, _np.inf, None
        for i in range(int(self.max_iter)):
            gradient = jacob(theta)
            g_norm = _np.linalg.norm(gradient, self.g_norm)
            if g_norm < self.g_tol:
                norm_inst = _base.NormValue(self.g_norm, g_norm)
                return _base.OptimizeResult(theta, norm_inst, i, True)
            g_square = gradient @ gradient
            beta = g_square / g_square_pre
            p = - gradient + beta * p
            theta += self.l_rate * p
            g_square_pre = g_square
        norm_inst = _base.NormValue(self.g_norm, g_norm)
        return _base.OptimizeResult(theta, norm_inst, self.max_iter, False)


class ConjugateGradientGenerator(_base.OptimizerGenerator):
    __slots__ = ['l_rate', 'max_iter', 'iter_mul']
    __default__ = {
        'l_rate': _learning_rate,
        'max_iter': _max_iteration,
        'iter_mul': _iteration_multiplier,
    }

    def __init__(self, l_rate: (float, int) = None, max_iter: (float, int) = None,
                 iter_mul: int = None):
        super().__init__(l_rate, max_iter, iter_mul)

    def get(self, loss: callable, init_val: _np.array,
            jacob: callable = None, hess: callable = None,
            ) -> _base.OptimizeGStep:
        i = 0
        theta_pre = init_val.copy()
        theta = theta_pre.copy()
        yield _base.OptimizeGStep(theta, i, theta_pre)
        p, g_square_pre = 0, _np.inf
        while i < self.max_iter:
            for j in range(self.iter_mul):
                gradient = jacob(theta)
                g_square = gradient @ gradient
                beta = g_square / g_square_pre
                p = - gradient + beta * p
                theta += self.l_rate * p
                g_square_pre = g_square
            i += self.iter_mul
            yield _base.OptimizeGStep(theta, i, theta_pre)
            theta_pre = theta.copy()


if __name__ == '__main__':
    pass
