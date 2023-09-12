import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score


def elastic_net(
        train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
        n_alphas: int) -> tuple[float, float, ElasticNetCV()]:
    # see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html
    l1_ratio = [.1, .5, .7, .9, .95, .99, 1]

    en = ElasticNetCV(l1_ratio=l1_ratio, n_alphas=n_alphas)
    en.fit(train_x, train_y)

    test_ybar = en.predict(test_x)
    r = np.corrcoef(test_y, test_ybar)[0, 1]
    cod = en.score(test_x, test_y)

    return r, cod, en


def kernel_ridge_corr(
        train_x: np.ndarray, train_y: np.ndarray, lambda_val: float, test_x: np.ndarray,
        test_y: np.ndarray) -> tuple[float, ...]:
    k_lambda = train_x + lambda_val * np.eye(train_x.shape[0])
    one_row = np.ones((train_x.shape[0], 1))

    # assuming N(features) > N(subjects)
    b_scalar = (
            np.linalg.solve(one_row.T @ np.linalg.solve(k_lambda, one_row), one_row.T)
            @ np.linalg.solve(k_lambda, train_y))
    alpha = np.linalg.solve(k_lambda, train_y.reshape(one_row.shape) - one_row * b_scalar)

    test_ybar = test_x @ alpha + np.ones((test_y.shape[0], 1)) * b_scalar
    r = np.corrcoef(test_y, test_ybar.T)[0, 1]

    return r, r2_score(test_y, test_ybar)


def kernel_ridge_corr_cv(
        train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
        test_y: np.ndarray) -> tuple[float, ...]:
    lambdas = [0, .0001, .0005, .001, .005, .01, .05, .1, .5, 1, 5, 10]
    kernel_x = np.corrcoef(np.vstack((train_x, test_x)))
    train_x_kernel = kernel_x[np.ix_(range(train_x.shape[0]), range(train_x.shape[0]))]
    test_x_kernel = kernel_x[
        np.ix_(range(train_x.shape[0], kernel_x.shape[0]), range(train_x.shape[0]))]

    r_lambdas = np.zeros(len(lambdas))
    for i, lambda_curr in enumerate(lambdas):
        rskf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=None)
        for train_ind, test_ind in rskf.split(train_x_kernel):
            r_curr, _ = kernel_ridge_corr(
                train_x_kernel[np.ix_(train_ind, train_ind)], train_y[train_ind], lambda_curr,
                train_x_kernel[np.ix_(test_ind, train_ind)], train_y[test_ind])
            r_lambdas[i] = r_lambdas[i] + r_curr / 10

    lambda_best = lambdas[np.argmax(r_lambdas)]
    r, cod = kernel_ridge_corr(train_x_kernel, train_y, lambda_best, test_x_kernel, test_y)

    return r, cod


def permutation_test(acc: np.ndarray, null_acc: np.ndarray) -> np.ndarray:
    all_acc = np.vstack((acc, null_acc))
    rank = all_acc.argsort(axis=0)
    ind = [rank.shape[0] - np.where(rank[:, i] == 0)[0][0] for i in range(rank.shape[1])]
    p = np.divide(ind, all_acc.shape[0])

    return p
