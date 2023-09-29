from typing import Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score


class KernelRidgeCorr(BaseEstimator):
    """Kernel Ridge Regression with correlation kernel"""

    def __init__(self, lambda_val: float = 0) -> None:
        self.lambda_val = lambda_val
        self.intercept_ = None
        self.coef_ = None
        self.x_train_ = None

    def fit(self, x_train: np.ndarray, y_train: Union[np.ndarray, None]) -> BaseEstimator:
        self.x_train_, y_checked = check_X_y(x_train, y_train)
        x_kernel = np.corrcoef(self.x_train_)
        k_lambda = x_kernel + self.lambda_val * np.eye(x_kernel.shape[0])
        one_row = np.ones((x_kernel.shape[0], 1))

        try:
            self.intercept_ = (
                    np.linalg.solve(one_row.T @ np.linalg.solve(k_lambda, one_row), one_row.T)
                    @ np.linalg.solve(k_lambda, y_checked))
        except np.linalg.LinAlgError:
            self.intercept_ = (
                    np.linalg.lstsq(one_row.T @ np.linalg.lstsq(k_lambda, one_row), one_row.T)
                    @ np.linalg.lstsq(k_lambda, y_checked))
        try:
            self.coef_ = np.linalg.solve(
                k_lambda, y_checked.reshape(one_row.shape) - one_row * self.intercept_)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(
                k_lambda, y_checked.reshape(one_row.shape) - one_row * self.intercept_)

        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        x_checked = check_array(x_test)

        x_kernel = np.corrcoef(np.vstack((self.x_train_, x_checked)))
        x_test_kernel = x_kernel[np.ix_(
            range(self.x_train_.shape[0], self.x_train_.shape[0] + x_checked.shape[0]),
            range(self.x_train_.shape[0]))]
        y_pred = x_test_kernel @ self.coef_ + np.ones((x_test_kernel.shape[0], 1)) * self.intercept_

        return y_pred

    def score(self, x_test: np.ndarray, y_test: np.ndarray, sample_weight=None):
        y_test_pred = self.predict(x_test)
        r2 = r2_score(y_test, y_test_pred, sample_weight=sample_weight)

        return r2
