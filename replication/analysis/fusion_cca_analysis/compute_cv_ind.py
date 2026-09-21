from pathlib import Path
import argparse

from sklearn.model_selection import RepeatedKFold
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description="Compute cross-validation indices for downstream analyses",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--dataset", type=str, dest="dataset", required=True, help="Dataset")
parser.add_argument(
    "--data_dir", type=Path, dest="data_dir", required=True, help="Absolute path to collected data")
parser.add_argument(
    "--out_dir", type=Path, dest="out_dir", required=True, help="Absolute path to output directory")
parser.add_argument(
    "--hcpya_res", type=str, dest="hcpya_res", default="", help="HCP-YA restricted data file")
args = parser.parse_args()

# Set-up
args.out_dir.mkdir(parents=True, exist_ok=True)
cv_seed = 42
n_repeats = 10
n_folds = 10

# Collected data for all subjects
x = pd.read_csv(Path(args.data_dir, f"{args.dataset}_x_arr.csv"), header=0, index_col=[0])
subjects = x.index.to_list()

# Cross-validation splits
if args.dataset == "HCP-YA":
    fam_id = pd.read_csv(args.hcpya_res, usecols=["Subject", "Family_ID"])
    fam_id = fam_id.loc[fam_id["Subject"].isin(subjects)]
    rng = np.random.default_rng(seed=cv_seed)
    cv_iter = [[[], []] for i in range(n_repeats * n_folds)]
    fold_size_min = np.round(len(subjects) / n_folds)
    for repeat in range(n_repeats):
        ind_to_fill = np.arange(len(subjects))
        for fold in range(n_folds):
            cv_ind = fold + repeat * n_folds
            n_max = len(subjects) - (fold + 1) * fold_size_min
            while len(ind_to_fill) > n_max and len(ind_to_fill):
                fill_start = rng.integers(low=0, high=len(ind_to_fill))
                fill_start_ind = ind_to_fill[fill_start]
                cv_iter[cv_ind][1].append(fill_start_ind)
                ind_to_fill = np.delete(ind_to_fill, fill_start)

                fill_fam_id = fam_id["Family_ID"].iloc[fill_start_ind]
                fill_fam = fam_id["Subject"].loc[
                    (fam_id["Family_ID"] == fill_fam_id) & (fam_id.index != fill_start_ind)]
                for ind in fill_fam.index.to_list():
                    cv_iter[cv_ind][1].append(ind)
                    ind_to_fill = np.delete(ind_to_fill, np.where(ind_to_fill == ind))
            cv_iter[cv_ind][0] = [
                i for i in range(len(subjects)) if i not in cv_iter[cv_ind][1]]
else:
    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=cv_seed)
    cv_iter = rkf.split(subjects)

# Save cross-validation indices
for fold, (train_ind, test_ind) in enumerate(cv_iter):
    train_out = Path(args.out_dir, f"{args.dataset}_train_fold{fold}.csv")
    pd.Series(train_ind).to_csv(train_out, header=False, index=False)
    test_out = Path(args.out_dir, f"{args.dataset}_test_fold{fold}.csv")
    pd.Series(test_ind).to_csv(test_out, header=False, index=False)
