from pathlib import Path
from timeit import default_timer
import argparse

from mpp.utilities import pheno_reg_conf
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
import numpy as np
import pandas as pd


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
    description="Run fusion CCA for region-wise features in cross-validation",
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

# Feature types
features_arr = ["rs_par", "s_myelin", "s_gmv"]
features_surf = ["s_cs", "s_ct"]
features_d = ["d_fa", "d_md", "d_ad", "d_rd"]

# Collected data for all subjects
x_arr = pd.read_csv(Path(args.data_dir, f"{args.dataset}_x_arr.csv"), header=0, index_col=[0])
x_d = pd.read_csv(Path(args.data_dir, f"{args.dataset}_x_d.csv"), header=0, index_col=[0])
y = pd.read_csv(Path(args.data_dir, f"{args.dataset}_y.csv"), header=0, index_col=[0])
conf = pd.read_csv(Path(args.data_dir, f"{args.dataset}_conf.csv"), header=0, index_col=[0])
subjects = x_arr.index.to_list()

# Iterate through folds
for fold in range(n_folds*n_repeats):
    # Fold-wise files
    train_file = Path(args.data_dir, f"{args.dataset}_train_fold{fold}.csv")
    test_file = Path(args.data_dir, f"{args.dataset}_test_fold{fold}.csv")

    # Fold-wise data
    train_ind = pd.read_csv(train_file, header=None).squeeze().to_list()
    test_ind = pd.read_csv(test_file, header=None).squeeze().to_list()

    # Iterate through prediction targets
    for target, y_curr in y.items():
        print(f"Computing for fold {fold} {target}")
        time_0 = default_timer()
        train_y, test_y, train_sub, test_sub = get_target_ind(y_curr, conf, train_ind, test_ind)
        res_curr = {
                "Dataset": args.dataset, "Repeat": int(np.floor(fold / n_folds)),
                "Fold": int(fold % n_folds), "Target": target}

        # For region-wise features, iterate through brain regions
        out_file = Path(args.out_dir, f"fusion_cca_{args.dataset}_{target}_fold{fold}_region.csv")
        train_x = x_arr.loc[train_sub]
        test_x = x_arr.loc[test_sub]
        results = {}
        for region in range(350):
            features = features_arr.copy()
            cols = [f"{feature}_{region}" for feature in features_arr]
            if region < 300:
                features.extend(features_surf)
                cols = cols + [f"{feature}_{region}" for feature in features_surf]
            train_x_reg = train_x[cols]
            test_x_reg = test_x[cols]

            results[region] = res_curr | {"Region": region} | cca(
                train_x_reg, train_y, test_x_reg, test_y, features)
        pd.DataFrame(results).T.to_csv(out_file)
        
        # For DTI features, iterate through the parcels in the white matter atlas
        out_file = Path(args.out_dir, f"fusion_cca_{args.dataset}_{target}_fold{fold}_dti.csv")
        train_x = x_d.loc[train_sub]
        test_x = x_d.loc[test_sub]
        results = {}
        for region in range(47):
            cols = [f"{feature}_{region}" for feature in features_d]
            train_x_d = train_x[cols]
            test_x_d = test_x[cols]

            results[region] = res_curr | {"Region": region} | cca(
                train_x_d, train_y, test_x_d, test_y, features_d)
        pd.DataFrame(results).T.to_csv(out_file)

        time_1 = default_timer()
        print(f"Finished computing fold {fold} {target} in {time_1 - time_0:.2f}s")
