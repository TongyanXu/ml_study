# -*- coding: utf-8 -*-
"""doc string"""

import numpy as _np
import utils as _utils


def linear_regression_by_gd(x, y, gd_cfg=None):
    """linear regression by gradient descent"""
    n_sample, n_feature = x.shape

    def loss_function(theta):
        v = x @ theta.T - y
        return v @ v / 2 / n_sample

    def gradient_function(theta):
        v = x @ theta.T - y
        return v.T @ x / n_sample

    _gd_cfg = dict(starting_point=_np.zeros(n_feature))
    if gd_cfg:
        _gd_cfg.update(gd_cfg)

    res = _utils.gradient_descent(
        loss_function=loss_function,
        gradient_function=gradient_function,
        **_gd_cfg
    )
    return res.theta
