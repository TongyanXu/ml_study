# -*- coding: utf-8 -*-
"""regression models"""

import numpy as _np
import types as _types


class RegressionModel:
    """
    Regression Models
    call RegressionModel.train to train data
    call RegressionModel.result to get result
    call RegressionModel.get_trainer to get data trainer (generator)
    """

    def __init__(self, x_matrix: (_np.array, _np.matrix),
                 y_vector: _np.array):
        """
        parameters
        ----------
        x_matrix: matrix of x observations (m samples * n features)
            first column should be 1 (for most regressions)
        y_vector: vector of y observations (m samples)

        """
        self.x, self.y = x_matrix, y_vector
        self.n_sample, self.n_feature = self.x.shape
        self.result = None

    def hypothesis(self, x_matrix: (_np.array, _np.matrix),
                   theta: _np.array) -> _np.array:
        """
        regression model hypothesis function - h(x, theta)
        given parameter combination theta and x matrix
        calculate y predictions according to model hypothesis
        """
        raise NotImplementedError

    def loss_function(self, theta: _np.array) -> float:
        """
        regression model loss function - f(theta)
        return loss value under given theta
        """
        raise NotImplementedError

    def jacob_function(self, theta: _np.array) -> _np.array:
        """
        regression model jacobian (gradient) function - g(theta)
        return gradient vector of given theta
        """
        raise NotImplementedError

    def hessian_function(self, theta: _np.array) -> _np.array:
        """
        regression model hessian function - h(theta)
        return hessian matrix of given theta
        """
        raise NotImplementedError

    def train(self, optimizer, init_val: _np.array = None) -> None:
        """
        train regression model with given optimizer and initial value

        parameters
        ----------
        optimizer: should be imported from <optimizer module>
            should be sub-instance of optimizer.base.OptimizerBase
        init_val: initial value of optimization
            should match the dimension of feature
            if None is given, use np.zeros instead

        returns
        -------
        optimization result
        instance of optimizer.base.OptimizerResult
        features:
            theta: solution of theta
            g_norm: norm and final value of gradient
            iteration: iteration count
            converge: converge or not
        """
        if init_val is None:
            init_val = _np.zeros(self.n_feature)
        self.result = optimizer.optimize(
            loss=self.loss_function,
            init_val=init_val,
            jacob=self.jacob_function,
            hess=self.hessian_function,
        )

    def get_trainer(self, opt_generator, init_val: _np.array = None
                    ) -> _types.GeneratorType:
        """
        get trainer (generator) with given optimizer generator and initial value

        parameters
        ----------
        opt_generator: optimizer generator
            should be imported from <optimizer module>
            should be sub-instance of optimizer.base.OptimizerGenerator
        init_val: initial value of optimization
            should match the dimension of feature
            if None is given, use np.zeros instead

        returns
        -------
        optimization step generator
        generator result: instance of optimizer.base.OptimizerGStep
        features:
            theta: current theta
            iteration: current iteration count
            theta_pre: last theta
        """
        if init_val is None:
            init_val = _np.zeros(self.n_feature)
        return opt_generator.get(
            loss=self.loss_function,
            init_val=init_val,
            jacob=self.jacob_function,
            hess=self.hessian_function,
        )


class LinearRegression(RegressionModel):
    """Linear Regression Model"""

    def hypothesis(self, x_matrix: (_np.array, _np.matrix),
                   theta: _np.array) -> _np.array:
        return x_matrix @ theta

    def loss_function(self, theta: _np.array) -> float:
        r = self.hypothesis(self.x, theta) - self.y
        return r @ r / 2 / self.n_sample

    def jacob_function(self, theta: _np.array) -> _np.array:
        r = self.hypothesis(self.x, theta) - self.y
        return r @ self.x / self.n_sample

    def hessian_function(self, theta: _np.array) -> _np.array:
        raise NotImplementedError


class LogisticRegression(RegressionModel):
    """Logistic Regression Model"""

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + _np.exp(- z))

    def hypothesis(self, x_matrix: (_np.array, _np.matrix),
                   theta: _np.array) -> _np.array:
        return self.sigmoid(x_matrix @ theta)

    def loss_function(self, theta: _np.array) -> float:
        part_1 = self.y @ _np.log(self.hypothesis(self.x, theta))
        part_2 = (1 - self.y) @ _np.log(1 - self.hypothesis(self.x, theta))
        return - (part_1 + part_2) / self.n_sample

    def jacob_function(self, theta: _np.array) -> _np.array:
        r = self.hypothesis(self.x, theta) - self.y
        return r @ self.x / self.n_sample

    def hessian_function(self, theta: _np.array) -> _np.array:
        raise NotImplementedError


if __name__ == '__main__':
    import optimizer as _opt

    # data
    lin_d = _np.loadtxt('linear_data.txt', delimiter=',')
    lin_d = _np.insert(lin_d, 0, values=1, axis=1)
    lin_x = lin_d[:, :2]
    lin_y = lin_d[:, 2]

    log_d = _np.loadtxt('logistic_data.txt', delimiter=',')
    log_d = _np.insert(log_d, 0, values=1, axis=1)
    log_x = log_d[:, :3]
    log_y = log_d[:, 3]

    # optimizer
    gd = _opt.GradientDescent(
        l_rate=2E-2,
        max_iter=1E5,
        g_tol=1E-5,
        g_norm=2,
    )
    cg = _opt.ConjugateGradient(
        l_rate=2E-2,
        max_iter=1E5,
        g_tol=1E-5,
        g_norm=2,
    )

    # model
    lin_model = LinearRegression(lin_x, lin_y)
    lin_model.train(
        optimizer=cg,
    )
    print(lin_model.result)

    # log_model = LogisticRegression(log_x, log_y)
    # log_model.train(
    #     optimizer=cg,
    # )
    # print(log_model.result)

    # # generator
    # gd_gen = _opt.GradientDescentGenerator(
    #     l_rate=1E-2,
    #     max_iter=1E5,
    #     iter_mul=10,
    # )
    # cg_gen = _opt.ConjugateGradientScipyGenerator(
    #     max_iter=1E5,
    #     iter_mul=10,
    # )
    #
    # # model generator
    # lin_gd_gen = lin_model.get_trainer(
    #     opt_generator=gd_gen,
    # )
    # print(next(lin_gd_gen))
    # print(next(lin_gd_gen))
    # print(next(lin_gd_gen))
    # log_cg_gen = log_model.get_trainer(
    #     opt_generator=cg_gen,
    # )
    # print(next(log_cg_gen))
    # print(next(log_cg_gen))
    # print(next(log_cg_gen))
