# -*- coding: utf-8 -*-
"""doc string"""

import numpy as _np


def linear_regression_by_ne(x, y, ne_cfg=None):
    """linear regression by normal equation"""
    _ = ne_cfg
    x_mat = _np.mat(x)
    theta = (x_mat.T @ x_mat).I @ x_mat.T @ y
    return _np.array(theta)[0]
