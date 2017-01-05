from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

from data.datasets import create_dataset


class Regression:
    def __init__(self, xs=list(), ys=list()):
        self.xs = np.array(xs, dtype=np.float64)
        self.ys = np.array(ys, dtype=np.float64)

        mean_x = mean(self.xs)
        mean_x2 = mean(self.xs * self.xs)
        mean_y = mean(self.ys)
        mean_xy = mean(self.xs * self.ys)

        cov = mean_xy - mean_x * mean_y
        var = mean_x2 - mean_x * mean_x

        self.m = cov / var
        self.intercept = mean_y - self.m * mean_x
        self.regression_line = [self.m * x + self.intercept for x in self.xs]

    def predict(self, predict_x):
        return predict_x * self.m + self.intercept

    def r_squared(self):
        mean_y = mean(self.ys)
        mean_ys = [mean_y for x in self.ys]
        total_sum_of_squares = sum((self.ys - mean_ys) ** 2)
        residual_sum_of_squares = sum((self.ys - self.regression_line) ** 2)

        return 1. - (residual_sum_of_squares / total_sum_of_squares)

    def draw(self, predict_x=None):
        style.use('ggplot')

        plt.scatter(self.xs, self.ys, color="#003D72", label="data")
        plt.plot(self.xs, self.regression_line, label="regression line")
        plt.legend(loc=2)
        if predict_x:
            plt.scatter(predict_x, self.predict(predict_x), color="g", label="predict")
        plt.legend(loc=4)
        plt.show()


if __name__ == '__main__':
    # xs, ys = create_dataset(n=40, variance=40, step=2, correlation='pos')
    xs, ys = create_dataset(n=40, variance=5, step=2)
    regression = Regression(xs, ys)
    regression.draw(7)
    print(regression.r_squared())
    # print(regression.best_fit_slope())
