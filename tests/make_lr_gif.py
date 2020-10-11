# -*- coding: utf-8 -*-
"""test display.lr_anime"""

import numpy as np
import optimizer as opt


def gd_bundle(
        l_rate,
        max_iter,
        g_tol,
        g_norm,
        iter_mul,
):
    gd = opt.GradientDescent(
        l_rate=l_rate,
        max_iter=max_iter,
        g_tol=g_tol,
        g_norm=g_norm,
    )
    gdg = opt.GradientDescentGenerator(
        l_rate=l_rate,
        max_iter=max_iter,
        iter_mul=iter_mul,
    )
    return gd, gdg


def cg_bundle(
        max_iter,
        g_tol,
        g_norm,
        iter_mul,
):
    cg = opt.ConjugateGradientScipy(
        max_iter=max_iter,
        g_tol=g_tol,
        g_norm=g_norm,
    )
    cgg = opt.ConjugateGradientScipyGenerator(
        max_iter=max_iter,
        iter_mul=iter_mul,
    )
    return cg, cgg


def gd_fig(x_vector, y_vector, figure):
    gd_opt, gd_opt_g = gd_bundle(
        l_rate=2.4E-2,
        max_iter=1E3,
        g_tol=1E-5,
        g_norm=2,
        iter_mul=10,
    )
    return display.lr_anime.make_1d_animation(
        figure=figure,
        x_vector=x_vector,
        y_vector=y_vector,
        init_val=None,
        optimizer=gd_opt,
        optimizer_generator=gd_opt_g,
        x_label='Population of City in 10,000s',
        y_label='Profit in $10,000s',
        reset=True,
        reset_pause=3,
        interval=100,
        contour_center='middle',
    )


def cg_fig(x_vector, y_vector, figure):
    cg_opt, cg_opt_g = cg_bundle(
        max_iter=5,
        g_tol=1E-5,
        g_norm=2,
        iter_mul=1,
    )
    return display.lr_anime.make_1d_animation(
        figure=figure,
        x_vector=x_vector,
        y_vector=y_vector,
        init_val=None,
        optimizer=cg_opt,
        optimizer_generator=cg_opt_g,
        x_label='Population of City in 10,000s',
        y_label='Profit in $10,000s',
        reset=True,
        reset_pause=3,
        interval=1000,
        contour_center='middle',
    )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import display

    data_path = r'test_data\city_population_profit.txt'
    test_data = np.loadtxt(data_path, delimiter=',')
    x, y = test_data[:, 0], test_data[:, 1]
    fig = plt.figure(figsize=(8, 7))

    # animation = gd_fig(x, y, fig)
    # animation.save('gd.gif', writer='pillow')
    # plt.show()
    animation = cg_fig(x, y, fig)
    animation.save('cg.gif', writer='pillow')
    # plt.show()
