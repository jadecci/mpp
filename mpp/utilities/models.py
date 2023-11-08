import numpy as np
from sklearn.linear_model import ElasticNetCV, LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from mpp.interfaces.models import KernelRidgeCorr
from sklearn.model_selection import GridSearchCV


def elastic_net(
        train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
        n_alphas: int) -> tuple[float, float, np.ndarray, np.ndarray, float]:
    # see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html
    l1_ratio = [.1, .5, .7, .9, .95, .99, 1]

    en = ElasticNetCV(l1_ratio=l1_ratio, n_alphas=n_alphas)
    en.fit(train_x, train_y)

    train_ypred = en.predict(train_x)
    test_ypred = en.predict(test_x)
    r = np.corrcoef(test_y, test_ypred)[0, 1]
    cod = en.score(test_x, test_y)
    cod_train = en.score(train_x, train_y)

    return r, cod, train_ypred, test_ypred, cod_train


def kernel_ridge_corr_cv(
        train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
        test_y: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    lambdas = [0, .0001, .0005, .001, .005, .01, .05, .1, .5, 1, 5, 10]
    krc_cv = GridSearchCV(estimator=KernelRidgeCorr(), param_grid={'lambda_val': lambdas})
    krc_cv.fit(train_x, train_y)

    train_ypred = krc_cv.predict(train_x)
    test_ypred = krc_cv.predict(test_x)
    r = np.corrcoef(test_y, test_ypred.T)[0, 1]
    cod = krc_cv.score(test_x, test_y)

    return r, cod, train_ypred.T, test_ypred.T


def linear(
        train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
        test_y: np.ndarray) -> tuple[float, float]:
    lr = LinearRegression()
    lr.fit(train_x, train_y)

    test_ybar = lr.predict(test_x)
    r = np.corrcoef(test_y, test_ybar)[0, 1]
    cod = lr.score(test_x, test_y)

    return r, cod


def random_forest_cv(
        train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
        test_y: np.ndarray) -> tuple[float, float, np.ndarray]:
    params = {
        'n_estimators': np.linspace(100, 1000, 4, dtype=int),
        'min_samples_split': np.linspace(0.01, 0.05, 5),
        'max_features': np.linspace(
            np.floor(train_x.shape[1]/2), train_x.shape[1], train_x.shape[1], dtype=int)}
    rfr_cv = GridSearchCV(
        estimator=RandomForestRegressor(criterion='friedman_mse'), param_grid=params)
    rfr_cv.fit(train_x, train_y)

    test_ybar = rfr_cv.predict(test_x)
    r = np.corrcoef(test_y, test_ybar)[0, 1]
    cod = rfr_cv.score(test_x, test_y)

    return r, cod, rfr_cv.best_estimator_.feature_importances_


def lasso_cv(
        train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
        n_alphas: int) -> tuple[float, float, np.ndarray]:
    lcv = LassoCV(n_alphas=n_alphas)
    lcv.fit(train_x, train_y)

    test_ypred = lcv.predict(test_x)
    r = np.corrcoef(test_y, test_ypred)[0, 1]
    cod = lcv.score(test_x, test_y)

    return r, cod, lcv.coef_


def random_patches(
        train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
        test_y: np.ndarray) -> tuple[float, float]:
    params = {
        'n_estimators': np.linspace(100, 1000, 4, dtype=int),
        'max_samples': np.linspace(0.5, 0.8, 4),
        'max_features': np.linspace(500, 1000, 5, dtype=int)}
    lambdas = [0, .0001, .0005, .001, .005, .01, .05, .1, .5, 1, 5, 10]
    krc_cv = GridSearchCV(estimator=KernelRidgeCorr(), param_grid={'lambda_val': lambdas})
    rp = GridSearchCV(estimator=BaggingRegressor(estimator=krc_cv), param_grid=params)
    rp.fit(train_x, train_y)

    test_ybar = rp.predict(test_x)
    r = np.corrcoef(test_y, test_ybar)[0, 1]
    cod = rp.score(test_x, test_y)

    return r, cod


def permutation_test(acc: np.ndarray, null_acc: np.ndarray) -> np.ndarray:
    all_acc = np.vstack((acc, null_acc))
    rank = all_acc.argsort(axis=0)
    ind = [rank.shape[0] - np.where(rank[:, i] == 0)[0][0] for i in range(rank.shape[1])]
    p = np.divide(ind, all_acc.shape[0])

    return p
