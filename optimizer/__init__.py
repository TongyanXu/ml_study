# -*- coding: utf-8 -*-
"""optimizers"""

__all__ = ['GradientDescent', 'ConjugateGradient', 'ConjugateGradientScipy', 'SBS']

from .gradient_descent import GradientDescent, GradientDescentSBS as _SBSgd
from .conjugate_gradient import ConjugateGradient, ConjugateGradientSBS as _SBScg
from .scipy_wrapper import ConjugateGradientScipy, ConjugateGradientScipySBS as _SBScgSP


class SBS:
    """wrapper for SBS class"""
    GradientDescent = _SBSgd
    ConjugateGradient = _SBScg
    ConjugateGradientScipy = _SBScgSP


if __name__ == '__main__':
    pass
