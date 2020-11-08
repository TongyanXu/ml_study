# -*- coding: utf-8 -*-
"""line search function"""

import numpy as np


def gold_stein_search(f, g, direction, position, max_alpha, rho, t):
    """..."""
    fk = f(position)
    gk = g(position)

    _min_alpha = 0
    _max_alpha = max_alpha

    phi0 = fk
    dphi0 = gk @ direction
    alpha = _max_alpha * np.random.uniform(0, 1)

    while True:
        phi = f(position + alpha * direction)
        if (phi - phi0) <= (rho * alpha * dphi0):
            if (phi - phi0) >= ((1 - rho) * alpha * dphi0):
                break
            else:
                _min_alpha = alpha
                if _max_alpha < max_alpha:
                    alpha = (_min_alpha + _max_alpha) / 2
                else:
                    alpha = t * alpha
        else:
            _max_alpha = alpha
            alpha = (_min_alpha + _max_alpha) / 2
    return alpha


def wolfe_search(f, g, direction, position, max_alpha, rho, t):
    '''
    线性搜索子函数
    数f，导数df，当前迭代点x和当前搜索方向d
    σ∈(ρ,1)=0.75
    '''
    sigma = 0.75

    flag = 0

    a = 0
    b = max_alpha
    fk = f(position)
    gk = g(position)

    phi0 = fk
    dphi0 = np.dot(gk, direction)
    alpha = b * np.random.uniform(0, 1)

    while (flag == 0):
        newfk = f(position + alpha * direction)
        phi = newfk
        # print(phi,phi0,rho,alpha ,dphi0)
        if (phi - phi0) <= (rho * alpha * dphi0):
            # if abs(np.dot(df(x + alpha * d),d))<=-sigma*dphi0:
            if (phi - phi0) >= ((1 - rho) * alpha * dphi0):
                flag = 1
            else:
                a = alpha
                b = b
                if (b < max_alpha):
                    alpha = (a + b) / 2
                else:
                    alpha = t * alpha
        else:
            a = a
            b = alpha
            alpha = (a + b) / 2
    return alpha


if __name__ == '__main__':
    pass
