# -*- coding: utf-8 -*-
"""optimizer base classes"""

import numpy as _np
import utils as _utils


class NormValue(_utils.FastInstance):
    __slots__ = ['norm', 'value']
    __display__ = __slots__


class OptimizeResult(_utils.FastInstance):
    __slots__ = ['theta', 'g_norm', 'iteration', 'converge']
    __display__ = ['iteration', 'g_norm', 'converge']


class OptimizeGStep(_utils.FastInstance):
    __slots__ = ['theta', 'iteration', 'theta_pre']
    __display__ = ['iteration']


class OptimizerBase(_utils.FastInstance):

    def optimize(self, loss: callable, init_val: _np.array,
                 jacob: callable = None, hess: callable = None) -> OptimizeResult:
        raise NotImplementedError


class OptimizerGenerator(_utils.FastInstance):

    def get(self, loss: callable, init_val: _np.array,
            jacob: callable = None, hess: callable = None) -> OptimizeGStep:
        raise NotImplementedError


if __name__ == '__main__':
    pass
