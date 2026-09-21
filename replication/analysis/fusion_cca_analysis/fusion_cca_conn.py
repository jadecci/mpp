from pathlib import Path
from timeit import default_timer
import argparse

from mpp.utilities import pheno_reg_conf
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
import numpy as np
import pandas as pd


task_list = {
    "HCP-YA": [
        "tfMRI_EMOTION", "tfMRI_GAMBLING", "tfMRI_LANGUAGE", "tfMRI_MOTOR", "tfMRI_WM",
        "tfMRI_RELATIONAL", "tfMRI_SOCIAL"],
    "HCP-A": ["tfMRI_CARIT_PA", "tfMRI_FACENAME_PA", "tfMRI_VISMOTOR_PA"],
    "HCP-D": ["tfMRI_CARIT", "tfMRI_EMOTION", "tfMRI_GUESSING"]}


def get_target_ind(
        target_y: pd.Series, conf_all: pd.DataFrame, train_ind: list, test_ind: list) -> [
            pd.Series, pd.Series, list, list]:
    train_target = target_y.iloc[train_ind]
    test_target = target_y.iloc[test_ind]
    train_conf = conf_all.iloc[train_ind]
    test_conf = conf_all.iloc[test_ind]

    # Only subjects with all data
    train_i = list(set(train_target.dropna().index) & set(train_conf.dropna(how="any").index))
    test_i = list(set(test_target.dropna().index) & set(test_conf.dropna(how="any").index))

    train_target, test_target = pheno_reg_conf(
                target_y.loc[train_i], conf_all.loc[train_i], target_y.loc[test_i],
                conf_all.loc[test_i])
    return train_target, test_target, train_i, test_i


def cca(
        x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame,
        y_test: np.ndarray, feature_names: list) -> dict:
    model = CCA(n_components=1)
    model.fit(x_train, y_train)

    x_scores, y_scores = model.transform(x_test, y_test)
    r, p = pearsonr(x_scores.flatten(), y_scores.flatten())

    cca_res = {"R": r, "P-value": p} | {
        f"{feature}_loading": load for load, feature in zip(model.x_loadings_, feature_names)}
    return cca_res


parser = argparse.ArgumentParser(
    description="Run fusion CCA for connectivity features in cross-validation",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--dataset", type=str, dest="dataset", required=True, help="Dataset")
parser.add_argument(
    "--data_dir", type=Path, dest="data_dir", required=True, help="Absolute path to collected data")
parser.add_argument(
    "--out_dir", type=Path, dest="out_dir", required=True, help="Absolute path to output directory")
args = parser.parse_args()

# Set-up
args.out_dir.mkdir(parents=True, exist_ok=True)
n_repeats = 10
n_folds = 10
features_ac = ["s_acgmv", "s_accs", "s_acct"]

# Feature types
features_sym = ["rs_sfc"] + [f"{task}_sfc" for task in task_list[args.dataset]]
features_asym = (
        ["rs_dfc", "rs_ec", "d_scc", "d_scl"] + [f"{task}_ec" for task in task_list[args.dataset]])

# Collected data for all subjects
x = pd.read_csv(Path(args.data_dir, f"{args.dataset}_x_conn.csv"), header=0, index_col=[0])
y = pd.read_csv(Path(args.data_dir, f"{args.dataset}_y.csv"), header=0, index_col=[0])
conf = pd.read_csv(Path(args.data_dir, f"{args.dataset}_conf.csv"), header=0, index_col=[0])
subjects = x.index.to_list()

for fold in range(n_repeats*n_folds):
    # Fold-wise files
    train_file = Path(args.data_dir, f"{args.dataset}_train_fold{fold}.csv")
    test_file = Path(args.data_dir, f"{args.dataset}_test_fold{fold}.csv")
    ac_files = {
        ac: Path(args.data_dir, f"{args.dataset}_{ac}_fold{fold}.csv") for ac in features_ac}

    # Fold-wise data
    train_ind = pd.read_csv(train_file, header=None).squeeze().to_list()
    test_ind = pd.read_csv(test_file, header=None).squeeze().to_list()
    x_ac = {ac: pd.read_csv(ac_files[ac], header=0, index_col=[0]) for ac in features_ac}

    # Iterate through prediction targets
    for target, y_curr in y.items():
        out_file = Path(args.out_dir, f"fusion_cca_{args.dataset}_{target}_fold{fold}_conn.csv")
        if not out_file.exists():
            print(f"Computing for fold {fold} {target}")
            time_0 = default_timer()
            train_y, test_y, train_sub, test_sub = get_target_ind(y_curr, conf, train_ind, test_ind)
            train_x = x.loc[train_sub]
            test_x = x.loc[test_sub]

            # Iterate through edges
            results = {}
            edge = 0
            for i in range(350):
                for j in range(i+1, 350):
                    res_curr = {
                        "Dataset": args.dataset, "Repeat": np.floor(fold / n_folds),
                        "Fold": fold % n_folds, "Target": target}

                    edge_opp = 350 * 350 - 1 - edge
                    features = features_sym + features_asym + ["s_acgmv"]
                    cols = (
                        [f"{feature}_{edge}" for feature in (features_sym + features_asym)]
                        + [f"{feature}_{edge_opp}" for feature in features_asym])
                    cols_gmv = [f"s_acgmv_{edge}", f"s_acgmv_{edge_opp}"]
                    train_x_edge = [train_x[cols], x_ac["s_acgmv"][cols_gmv].loc[train_sub]]
                    test_x_edge = [test_x[cols], x_ac["s_acgmv"][cols_gmv].loc[test_sub]]

                    # only include cortical features for edges between cortical regions
                    if i < 300 and j < 300:
                        edge_opp_surf = 300 * 300 - 1 - edge
                        for feature in ["s_accs", "s_acct"]:
                            features.extend(["s_accs", "s_acct"])
                            cols = [f"{feature}_{edge}", f"{feature}_{edge_opp_surf}"]
                            train_x_edge.append(x_ac[feature][cols].loc[train_sub])
                            test_x_edge.append(x_ac[feature][cols].loc[test_sub])

                    train_x_edge = pd.concat(train_x_edge, axis="columns")
                    test_x_edge = pd.concat(test_x_edge, axis="columns")

                    results[edge] = res_curr | {"Edge": edge} | cca(
                        train_x_edge, train_y, test_x_edge, test_y, features)
                    edge += 1

            pd.DataFrame(results).T.to_csv(out_file)
            time_1 = default_timer()
            print(f"Finished computing fold {fold} {target} in {time_1 - time_0:.2f}s")
