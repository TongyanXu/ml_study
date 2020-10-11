# -*- coding: utf-8 -*-
"""optimizers using scipy"""

import numpy as _np
import scipy.optimize as _opt
import optimizer.base as _base

_max_iteration = 1E5
_g_tolerance = 1E-5
_g_norm = 2
_iteration_multiplier = 1


class ConjugateGradientScipy(_base.OptimizerBase):
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
        callback_info = {'iter': 0}

        def callback(xk):
            callback_info['iter'] += 1

        theta, _, _, _, flag = _opt.fmin_cg(
            f=loss,
            x0=theta,
            fprime=jacob,
            gtol=self.g_tol,
            norm=self.g_norm,
            maxiter=self.max_iter,
            full_output=True,
            disp=False,
            callback=callback,
        )
        gradient = jacob(theta)
        g_norm = _np.linalg.norm(gradient, self.g_norm)
        norm_inst = _base.NormValue(self.g_norm, g_norm)
        iteration = callback_info['iter']
        converge = flag == 0
        return _base.OptimizeResult(theta, norm_inst, iteration, converge)


class ConjugateGradientScipyGenerator(_base.OptimizerGenerator):
    __slots__ = ['max_iter', 'iter_mul']
    __default__ = {
        'max_iter': _max_iteration,
        'iter_mul': _iteration_multiplier,
    }

    def __init__(self, max_iter: (float, int) = None, iter_mul: int = None):
        super().__init__(max_iter, iter_mul)

    def get(self, loss: callable, init_val: _np.array,
            jacob: callable = None, hess: callable = None,
            ) -> _base.OptimizeGStep:
        theta = init_val.copy()
        last_theta = theta.copy()

        theta, theta_hist = _opt.fmin_cg(
            f=loss,
            x0=theta,
            fprime=jacob,
            gtol=0.0,
            maxiter=self.max_iter,
            disp=False,
            retall=True,
        )
        i = 0
        while i <= self.max_iter:
            if i < len(theta_hist):
                this_theta = theta_hist[i]
                yield _base.OptimizeGStep(this_theta, i, last_theta)
                last_theta = this_theta
            else:
                yield _base.OptimizeGStep(theta, i, theta)
            i += self.iter_mul


if __name__ == '__main__':
    pass
