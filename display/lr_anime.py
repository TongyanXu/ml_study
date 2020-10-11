# -*- coding: utf-8 -*-
"""animation for linear regression"""

import matplotlib.animation as _anime
import numpy as _np
import models as _models


def _extend(x1, x2, factor=4):
    return x1 * (factor + 1) / factor - x2 / factor


def make_1d_animation(figure, x_vector, y_vector, optimizer, optimizer_generator,
                      init_val=None, x_label='X', y_label='Y',
                      reset=True, reset_pause=30, interval=100, contour_center='solution'):

    init_val = _np.zeros(2)
    x_matrix = _np.array([_np.ones(len(x_vector)), x_vector]).T
    model = _models.LinearRegression(x_matrix, y_vector)
    model.train(optimizer=optimizer, init_val=init_val)

    class _ScatterPlot:

        def __init__(self, axes):
            self._axes = axes
            self._scatter = self._axes.scatter(
                x_vector, y_vector, s=10, c='red',
                marker='x', label='Training data',
            )
            self._line, = self._axes.plot(
                [min(x_vector), max(x_vector)], [0, 0],
                c='blue', label='Linear regression',
            )
            self._axes.set_xlabel(x_label)
            self._axes.set_ylabel(y_label)
            self._axes.set_xlim((min(x_vector), max(x_vector)))
            self._axes.set_ylim((min(y_vector), max(y_vector)))
            self._axes.legend(loc='lower right')

        def update(self, gd_res):
            y0 = gd_res.theta @ _np.array([1, min(x_vector)])
            y1 = gd_res.theta @ _np.array([1, max(x_vector)])
            self._line.set_ydata([y0, y1])

        def reset(self):
            pass

    class _ContourPlot:

        def __init__(self, axes):
            self._axes = axes
            theta0, theta1, levels = self._contour_range()
            loss = _np.array([[model.loss_function(_np.array([t0, t1]))
                               for t0 in theta0] for t1 in theta1])
            x_theta, y_theta = _np.meshgrid(theta0, theta1)
            self._contour = self._axes.contour(x_theta, y_theta, loss,
                                               levels=levels)
            self._axes.set_xlabel('Theta 0')
            self._axes.set_ylabel('Theta 1')

        @staticmethod
        def _contour_range():
            theta0_beg = _extend(init_val[0], model.result.theta[0])
            theta1_beg = _extend(init_val[1], model.result.theta[1])
            theta0_end = _extend(model.result.theta[0], init_val[0])
            theta1_end = _extend(model.result.theta[1], init_val[1])

            if contour_center == 'solution':
                theta0_end += model.result.theta[0] - init_val[0]
                theta1_end += model.result.theta[1] - init_val[1]

            theta0 = _np.linspace(theta0_beg, theta0_end, 101)
            theta1 = _np.linspace(theta1_beg, theta1_end, 101)

            theta0_lv_beg = _extend(theta0_beg, theta0_end, 3)
            theta0_lv_end = _extend(theta0_end, theta0_beg, 3)
            theta1_lv_beg = _extend(theta1_beg, theta1_end, 3)
            theta1_lv_end = _extend(theta1_end, theta1_beg, 3)

            theta0_lv = _np.linspace(theta0_lv_beg, theta0_lv_end, 16)
            theta1_lv = _np.linspace(theta1_lv_beg, theta1_lv_end, 16)

            levels = [model.loss_function(_np.array([t0, t1]))
                      for t0, t1 in zip(theta0_lv, theta1_lv)]
            return theta0, theta1, sorted(levels)

        def update(self, gd_res):
            x = [gd_res.theta_pre[0], gd_res.theta[0]]
            y = [gd_res.theta_pre[1], gd_res.theta[1]]
            self._axes.plot(x, y, c='red')

        def reset(self):
            self._axes.lines = [self._axes.lines[0]]

    class _LinePlot:

        def __init__(self, axes):
            self._axes = axes
            self._axes.set_xlabel('Iteration')
            self._axes.set_ylabel('Loss')
            self._axes.set_xlim((0, optimizer_generator.max_iter))
            self._axes.set_ylim((0, model.loss_function(init_val)))

        def update(self, gd_res):
            iter_p = max(gd_res.iteration - optimizer_generator.iter_mul, 0)
            x = [iter_p, gd_res.iteration]
            y = [model.loss_function(gd_res.theta_pre),
                 model.loss_function(gd_res.theta)]
            self._axes.plot(x, y, c='green')

        def reset(self):
            self._axes.lines = [self._axes.lines[0]]

    class _Plots:

        def __init__(self, fig):
            self._plots = [
                _ScatterPlot(fig.add_subplot(211)),
                _ContourPlot(fig.add_subplot(223)),
                _LinePlot(fig.add_subplot(224)),
            ]

        def update(self, gd_res):
            _ = [plot.update(gd_res) for plot in self._plots]

        def reset(self):
            _ = [plot.reset() for plot in self._plots]

    anime_plot = _Plots(figure)
    global_param = {
        'generator': model.get_trainer(
            opt_generator=optimizer_generator,
            init_val=init_val,
        ),
        'reset_count': 0,
    }

    def animate(i):
        generator = global_param['generator']
        try:
            res = next(generator)
        except StopIteration:
            if reset:
                global_param['reset_count'] += 1
                if global_param['reset_count'] > reset_pause:
                    global_param['reset_count'] = 0
                    anime_plot.reset()
                    global_param['generator'] = model.get_trainer(
                        opt_generator=optimizer_generator,
                        init_val=init_val,
                    )
        else:
            anime_plot.update(res)

    return _anime.FuncAnimation(fig=figure, func=animate, interval=interval)
