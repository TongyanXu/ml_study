# -*- coding: utf-8 -*-
"""doc string"""

from utils import gradient_descent as _gd

_default_gd_cfg = dict(
    learning_rate=1E-2,
    precision=1E-16,
    max_iter=1E5,
    starting_point=None,
)


def linear_regression_by_gd(x, y, gd_cfg=None):
    """linear regression by gradient descent"""
    n_sample, n_feature = x.shape

    def loss_function(theta):
        v = x @ theta.T - y
        return v @ v / 2 / n_sample

    def gradient_function(theta):
        v = x @ theta.T - y
        return v.T @ x / n_sample

    _gd_cfg = gd_cfg or _default_gd_cfg
    return _gd.simple_gradient_descent(
        loss_function=loss_function,
        gradient_function=gradient_function,
        variable_count=n_feature,
        **_gd_cfg
    )
