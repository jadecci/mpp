from pathlib import Path

from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np


def find_sub_file(sublists: dict, features_dir: Path, subject: str) -> Path:
    dataset = "HCP-YA"
    for key, val in sublists.items():
        if subject in val:
            dataset = key
    dataset_dir = Path(features_dir, dataset)
    if dataset in ["HCP-A", "HCP-D"]:
        subject_file = Path(dataset_dir, f"{subject}_V1_MR.h5")
    else:
        subject_file = Path(dataset_dir, f"{subject}.h5")
    return subject_file


def elastic_net(
        train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
        n_alphas: int) -> tuple[float, float, np.ndarray, float]:
    # see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html
    l1_ratio = [.1, .5, .7, .9, .95, .99, 1]

    en = ElasticNetCV(l1_ratio=l1_ratio, n_alphas=n_alphas)
    en.fit(train_x, train_y)

    ypred = en.predict(test_x)
    r = np.corrcoef(test_y, ypred)[0, 1]
    cod = en.score(test_x, test_y)
    train_cod = en.score(train_x, train_y)

    return r, cod, ypred, train_cod


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


def feature_list(dataset: str) -> tuple[list, ...]:
    base_feature_list = [
        "rs_sfc", "rs_dfc", "rs_ec", "rs_stats", "s_myelin", "s_gmv", "s_cs", "s_ct", "d_scc",
        "d_scl", "d_fa", "d_md", "d_ad", "d_rd"]
    cv_feature_list = ["rs_grad", "s_acgmv", "s_accs", "s_acct"]
    task_feature_list = {
        "HCP-YA": [
            "tfMRI_EMOTION_sfc", "tfMRI_GAMBLING_sfc", "tfMRI_LANGUAGE_sfc", "tfMRI_MOTOR_sfc",
            "tfMRI_WM_sfc", "tfMRI_RELATIONAL_sfc", "tfMRI_SOCIAL_sfc",
            "tfMRI_EMOTION_ec", "tfMRI_GAMBLING_ec", "tfMRI_LANGUAGE_ec", "tfMRI_MOTOR_ec",
            "tfMRI_WM_ec", "tfMRI_RELATIONAL_ec", "tfMRI_SOCIAL_ec"],
        "HCP-A": [
            "tfMRI_CARIT_PA_sfc", "tfMRI_FACENAME_PA_sfc", "tfMRI_VISMOTOR_PA_sfc",
            "tfMRI_CARIT_PA_ec", "tfMRI_FACENAME_PA_ec", "tfMRI_VISMOTOR_PA_ec"],
        "HCP-D": [
            "tfMRI_CARIT_sfc", "tfMRI_EMOTION_sfc", "tfMRI_GUESSING_sfc",
            "tfMRI_CARIT_ec", "tfMRI_EMOTION_ec", "tfMRI_GUESSING_ec"]}
    return base_feature_list, cv_feature_list, task_feature_list[dataset]
