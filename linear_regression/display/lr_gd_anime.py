# -*- coding: utf-8 -*-
"""doc string"""

import matplotlib.animation as _anime
import numpy as _np


class _GDResult:
    __slots__ = ['theta', 'loss', 'converge', 'iteration',
                 'theta_pre', 'loss_pre']
    __optional__ = ['theta_pre', 'loss_pre']

    def __init__(self, theta, loss, converge, iteration,
                 theta_pre, loss_pre):
        self.theta = theta
        self.loss = loss
        self.converge = converge
        self.iteration = iteration
        self.theta_pre = theta_pre
        self.loss_pre = loss_pre

    def __repr__(self):
        idt = [f'{attr}={getattr(self, attr)}'
               for attr in self.__slots__
               if attr not in self.__optional__]
        return f"[GD result - {', '.join(idt)}]"


def _gd_by_step(loss_function, gradient_function, variable_count,
                learning_rate=1E-2, iteration_multiplier=10,
                max_iteration=1E5, starting_point=None):
    i = 0
    theta_pre = _np.zeros(variable_count) \
        if starting_point is None else starting_point
    loss_pre = loss_function(theta_pre)
    yield _GDResult(theta_pre, loss_pre, 0., i,
                    theta_pre, loss_pre)
    theta = theta_pre.copy()
    while i < max_iteration:
        for j in range(iteration_multiplier):
            gradient = gradient_function(theta)
            theta -= learning_rate * gradient
        loss = loss_function(theta)
        converge = abs(loss - loss_pre)
        i += iteration_multiplier
        yield _GDResult(theta, loss, converge, i,
                        theta_pre, loss_pre)
        theta_pre, loss_pre = theta.copy(), loss.copy()


class LRGDByStep:

    def __init__(self, x_matrix, y_vector, learning_rate=1E-2,
                 iteration_multiplier=10, max_iteration=1E5,
                 starting_point=None):
        self._x, self._y = x_matrix, y_vector
        self._n_sample, self._n_feature = self._x.shape
        assert self._n_sample == len(self._y)
        self._learning_rate = learning_rate
        self._iteration_multiplier = iteration_multiplier
        self._max_iteration = max_iteration
        self._starting_point = starting_point
        self._gd = None
        self.reset()

    def _y_predict_diff(self, theta):
        return self._x @ theta.T - self._y

    def loss_function(self, theta):
        v = self._y_predict_diff(theta)
        return v @ v / 2 / self._n_sample

    def gradient_function(self, theta):
        v = self._y_predict_diff(theta)
        return v.T @ self._x / self._n_sample

    def next_step(self):
        try:
            return next(self._gd)
        except StopIteration:
            return None

    def reset(self):
        self._gd = _gd_by_step(
            loss_function=self.loss_function,
            gradient_function=self.gradient_function,
            variable_count=self._n_feature,
            learning_rate=self._learning_rate,
            iteration_multiplier=self._iteration_multiplier,
            max_iteration=self._max_iteration,
            starting_point=self._starting_point,
        )


class LRGDByStep1D(LRGDByStep):

    def __init__(self, x_vector, y_vector, learning_rate=1E-2,
                 iteration_multiplier=10, max_iteration=1E5,
                 starting_point=None):
        x_matrix = _np.array([_np.ones(len(x_vector)), x_vector]).T
        super().__init__(x_matrix, y_vector, learning_rate,
                         iteration_multiplier, max_iteration,
                         starting_point)


def make_animation(figure, x_vector, y_vector, x_label='X', y_label='Y',
                   learning_rate=1E-2, iteration_multiplier=10,
                   max_iteration=1E5, starting_point=None,
                   reset=True, reset_pause=30, interval=100):

    if starting_point is None:
        starting_point = _np.zeros(2)
    starting_point = starting_point.astype(float)

    model = LRGDByStep1D(
        x_vector=x_vector, y_vector=y_vector,
        learning_rate=learning_rate,
        iteration_multiplier=iteration_multiplier,
        max_iteration=max_iteration,
        starting_point=starting_point,
    )

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
            theta_0 = _np.arange(-10, 10.2, 0.2)
            theta_1 = _np.arange(-1, 4.05, 0.05)
            loss = _np.array([[model.loss_function(_np.array([t0, t1]))
                               for t0 in theta_0] for t1 in theta_1])
            x_theta, y_theta = _np.meshgrid(theta_0, theta_1)
            contour_levels = [0, 5, 10, 20, 40, 80, 160, 320, 640]
            self._contour = axes.contour(x_theta, y_theta, loss,
                                         levels=contour_levels)
            self._axes.set_xlabel('Theta 0')
            self._axes.set_ylabel('Theta 1')

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
            self._axes.set_xlim((0, max_iteration))
            self._axes.set_ylim((0, model.loss_function(starting_point)))

        def update(self, gd_res):
            iter_p = max(gd_res.iteration - iteration_multiplier, 0)
            x = [iter_p, gd_res.iteration]
            y = [gd_res.loss_pre, gd_res.loss]
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
    status_control = {'reset_count': 0}

    def animate(i):
        res = model.next_step()
        if res:
            anime_plot.update(res)
        elif reset:
            status_control['reset_count'] += 1
            if status_control['reset_count'] > reset_pause:
                status_control['reset_count'] = 0
                anime_plot.reset()
                model.reset()

    return _anime.FuncAnimation(fig=figure, func=animate, interval=interval)
