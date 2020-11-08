# -*- coding: utf-8 -*-
"""optimizer base classes"""

import numpy as _np
import utils as _utils


class NormValue(_utils.FastInstance):
    """special instance for saving norm value and order"""
    __slots__ = ['norm', 'value']
    __display__ = __slots__


class OptimizeResult(_utils.FastInstance):
    """optimizer result instance"""
    __slots__ = ['theta', 'g_norm', 'iteration', 'converge']
    __display__ = ['iteration', 'g_norm', 'converge']


class OptimizeStepResult(_utils.FastInstance):
    """optimizer step-by-step result for each step"""
    __slots__ = ['theta', 'iteration', 'theta_pre']
    __display__ = ['iteration']


class OptimizerBase(_utils.FastInstance):
    """
    base class for all optimizers
    optimizer is used to find local minimum of input loss functions
    """

    def optimize(self, loss: callable, init_val: _np.array,
                 jacob: callable = None, hess: callable = None) -> OptimizeResult:
        """
        to optimize loss function to local minimum

        parameters
        ----------
        loss: loss function whose input is numpy array of parameters
            returns single value indicating loss for input parameters
        init_val: initial value of parameters
            should be a numpy array and can be set as input of loss function
        jacob: jacob function whose input is same as loss function's
            returns a vector indicating 1st order gradient of each parameter
        hess: hessian function whose input is same as loss function's
            returns a matrix indicating 2nd order gradient of each parameter

        returns
        -------
        OptimizeResult instance
        """
        raise NotImplementedError


class OptimizerSBS(_utils.FastInstance):
    """
    step-by-step base class for all optimizers
    SBS instance can return snapshot of each iteration of optimizing process
    """

    def make(self, loss: callable, init_val: _np.array,
             jacob: callable = None, hess: callable = None) -> OptimizeStepResult:
        """
        make step-by-step generator

        parameters
        ----------
        loss: loss function whose input is numpy array of parameters
            returns single value indicating loss for input parameters
        init_val: initial value of parameters
            should be a numpy array and can be set as input of loss function
        jacob: jacob function whose input is same as loss function's
            returns a vector indicating 1st order gradient of each parameter
        hess: hessian function whose input is same as loss function's
            returns a matrix indicating 2nd order gradient of each parameter

        returns
        -------
        generator of OptimizeStepResult
        """
        raise NotImplementedError


if __name__ == '__main__':
    pass
