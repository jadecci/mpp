from pathlib import Path

from sklearn.linear_model import ElasticNetCV
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


def feature_list(datasets: list) -> list:
    base_feature_list = [
        "rs_sfc", "rs_dfc", "rs_ec", "rs_stats", "rs_grad", "s_myelin", "s_gmv", "s_cs", "s_ct",
        "s_acgmv", "s_accs", "s_acct", "d_scc", "d_scl", "d_fa", "d_md", "d_ad", "d_rd"]
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
    if len(datasets) == 1:
        return base_feature_list + task_feature_list[datasets[0]]
    else:
        return base_feature_list


def elastic_net(
        train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
        test_y: np.ndarray, n_alphas) -> tuple[float, float, np.ndarray]:
    # see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html # noqa: E501
    l1_ratio = [.1, .5, .7, .9, .95, .99, 1]
    en = ElasticNetCV(l1_ratio=l1_ratio, n_alphas=n_alphas)
    en.fit(train_x, train_y)
    ypred = en.predict(test_x)
    r = np.corrcoef(test_y, ypred)[0, 1]
    return r, en.score(test_x, test_y), ypred