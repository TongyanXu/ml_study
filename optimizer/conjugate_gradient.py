# -*- coding: utf-8 -*-
"""optimizer - conjugate descent"""

import numpy as _np
import optimizer.base as _base

_max_iteration = 1E5
_g_tolerance = 1E-5
_g_norm = 2
_iteration_multiplier = 1


class ConjugateGradient(_base.OptimizerBase):
    __slots__ = ['max_iter', 'g_tol', 'g_norm']
    __default__ = {
        'max_iter': _max_iteration,
        'g_tol': _g_tolerance,
        'g_norm': _g_norm,
    }

    def __init__(self, max_iter: (float, int) = None,
                 g_tol: (float, int) = None, g_norm: int = None):
        super().__init__(max_iter, g_tol, g_norm)
        self.max_iter = int(self.max_iter)

    def optimize(self, loss: callable, init_val: _np.array,
                 jacob: callable = None, hess: callable = None
                 ) -> _base.OptimizeResult:
        theta = init_val.copy()
        gradient = jacob(theta)
        g_norm = _np.linalg.norm(gradient, self.g_norm)
        p = - gradient
        for i in range(int(self.max_iter)):
            delta = g_norm ** 2
            hessian = hess(theta)
            alpha = (gradient @ p) / (p @ hessian @ p)
            theta -= alpha * gradient
            gradient = jacob(theta)
            g_norm = _np.linalg.norm(gradient, self.g_norm)
            if g_norm < self.g_tol:
                norm_inst = _base.NormValue(self.g_norm, g_norm)
                return _base.OptimizeResult(theta, norm_inst, i + 1, True)
            beta = (g_norm ** 2) / delta
            p = - gradient + beta * p
        norm_inst = _base.NormValue(self.g_norm, g_norm)
        return _base.OptimizeResult(theta, norm_inst, self.max_iter, False)


if __name__ == '__main__':
    pass
