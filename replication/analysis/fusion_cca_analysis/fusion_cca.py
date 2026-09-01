from pathlib import Path
import argparse

from mpp.utilities import pheno_reg_conf
from scipy.stats import pearsonr
from sklearn.decomposition import CCA
from sklearn.model_selection import RepeatedKFold
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd


def apply_ac_params(morph: pd.DataFrame, param: pd.DataFrame) -> pd.DataFrame:
    ac = []
    for i in range(nparc):
        for j in range(nparc):
            param_curr = param[f"{i}_{j}"]
            morph_curr = morph[f"{feature}_{i}"]
            ac.append((
                param_curr[0] + param_curr[1] * morph_curr + param_curr[2]
                * morph.mean(axis=1)).iloc[0])
    return pd.DataFrame(ac).T


def cca(
        x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame,
        y_test: np.ndarray) -> dict:
    model = CCA(n_components=1)
    model.fit(x_train, y_train)
    cca_res = {
        f"{feature}_loading": load for load, feature in zip(
            model.x_loadings_, model.feature_names_in)}

    x_scores, y_scores = model.transform(x_test, y_test)
    r, p = pearsonr(x_scores.flatten(), y_scores.flatten())
    cca_res = cca_res | {"R": r, "P-value": p}
    return cca_res


parser = argparse.ArgumentParser(
    description="Run fusion CCA in cross-validation",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument(
    "--data_dir", type=Path, dest="data_dir", required=True, help="Absolute path to collected data")
parser.add_argument(
    "--out_dir", type=Path, dest="out_dir", required=True, help="Absolute path to output directory")
parser.add_argument(
    "--hcpya_res", type=Path, dest="hcpya_res", required=True, help="HCP-YA restricted data file")
args = parser.parse_args()

# Set-up
args.out_dir.mkdir(parents=True, exist_ok=True)
cv_seed = 42
n_repeats = 10
n_folds = 10

# list of tasks in each dataset
task_list = {
    "HCP-YA": [
        "tfMRI_EMOTION", "tfMRI_GAMBLING", "tfMRI_LANGUAGE", "tfMRI_MOTOR", "tfMRI_WM",
        "tfMRI_RELATIONAL", "tfMRI_SOCIAL"],
    "HCP-A": ["tfMRI_CARIT_PA", "tfMRI_FACENAME_PA", "tfMRI_VISMOTOR_PA"],
    "HCP-D": ["tfMRI_CARIT", "tfMRI_EMOTION", "tfMRI_GUESSING"]}

# Fixed feature types
features_arr = ["rs_par", "s_myelin", "s_gmv"]
features_surf = ["s_cs", "s_ct"]
features_d = ["d_fa", "d_md", "d_ad", "d_rd"]

results = []
for dataset in ["HCP-D", "HCP-YA", "HCP-A"]:
    # Dataset-specific feature types
    features_sym = ["rs_sfc"] + [f"{task}_sfc" for task in task_list[dataset]]
    features_asym = (
            ["rs_dfc", "rs_ec", "d_scc", "d_scl"] + [f"{task}_ec" for task in task_list[dataset]])

    # Collected data for all subjects
    x = pd.read_csv(Path(args.data_dir, f"{dataset}_x.csv"), header=0, index_col=[0])
    y = pd.read_csv(Path(args.data_dir, f"{dataset}_y.csv"), header=0, index_col=[0])
    conf = pd.read_csv(Path(args.data_dir, f"{dataset}_conf.csv"), header=0, index_col=[0])
    subjects = x.index.to_list

    # Cross-validation splits
    if dataset == "HCP-YA":
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

    # Iterate through folds
    for fold, (train_ind, test_ind) in enumerate(cv_iter):
        train_x = x.iloc[train_ind]
        test_x = x.iloc[test_ind]

        # Add structural co-registration features
        x_ac = {}
        for feature in ["s_gmv", "s_cs", "s_ct"]:
            # Get features from training set
            nparc = 350 if feature == "s_gmv" else 300
            features = train_x.filter(regex=f"{feature}_*", axis="columns")
            features[features.columns] = features[features.columns].apply(pd.to_numeric)
            features.columns = range(nparc)
            features = features.join(pd.DataFrame({"mean": features.mean(axis=1)}))

            # Estimate paramaters
            params = pd.DataFrame()
            for i in range(nparc):
                for j in range(nparc):
                    res = ols(f"features[{i}] ~ features[{j}] + mean", data=features).fit()
                    params[f"{i}_{j}"] = [
                        res.params["Intercept"], res.params[f"features[{j}]"], res.params["mean"]]

            # Apply to all subjects
            ac_feature = []
            for subject in subjects:
                features_sub = x.loc[subject].filter(regex=f"{feature}_*", axis="columns")
                ac_curr = apply_ac_params(features_sub, params)
                ac_curr.columns = [
                    f"s_ac{feature.split('s_')[1]}_{col}" for col in range(len(ac_curr.columns))]
                ac_curr.index = [subject]
                ac_feature.append(ac_curr)
            x_ac[feature] = pd.concat(ac_feature, axis="index")

        # Iterate through prediction targets
        for target, y_curr in y.items():
            train_y, test_y = pheno_reg_conf(
                y_curr.iloc[train_ind], conf.iloc[train_ind], y_curr.iloc[test_ind],
                conf.iloc[test_ind])

            # For connectivity features, iterate through edges
            edge = 0
            for i in range(350):
                for j in range(i+1, 350):
                    res_curr = {
                        "Dataset": dataset, "Repeat": np.floor(fold / n_folds),
                        "Fold": fold % n_folds, "Target": target}

                    edge_opp = 350 * 350 - 1 - edge
                    cols = (
                        [f"{feature}_{edge}" for feature in (features_sym + features_asym)]
                        + [f"{feature}_{edge_opp}" for feature in features_asym])
                    cols_gmv = [f"s_acgmv_{edge}", f"s_acgmv_{edge_opp}"]
                    train_x_edge = [train_x[cols], x_ac["s_gmv"][cols_gmv].iloc[train_ind]]
                    test_x_edge = [test_x[cols], x_ac["s_gmv"][cols_gmv].iloc[test_ind]]

                    # only include cortical features for edges between cortical regions
                    if i < 300 and j < 300:
                        edge_opp_surf = 300 * 300 - 1 - edge
                        for feature in ["cs", "ct"]:
                            cols = [f"s_ac{feature}_{edge}", f"s_ac{feature}_{edge_opp_surf}"]
                            train_x_edge.append(x_ac[f"s_{feature}"][cols].iloc[train_ind])
                            test_x_edge.append(x_ac[f"s_{feature}"][cols].iloc[test_ind])

                    train_x_edge = pd.concat(train_x_edge, axis="columns")
                    test_x_edge = pd.concat(test_x_edge, axis="columns")
                    res_curr = res_curr | cca(train_x_edge, train_y, test_x_edge, test_y)

                    results.append(pd.DataFrame(res_curr, index=[f"edge_{edge}"]))
                    edge += 1

            # For region-wise features, iterate through brain regions
            for region in range(350):
                res_curr = {
                    "Dataset": dataset, "Repeat": np.floor(fold / n_folds),
                    "Fold": fold % n_folds, "Target": target}

                cols = [f"{feature}_{region}" for feature in features_arr]
                if region < 300:
                    cols = cols + [f"{feature}_{region}" for feature in features_surf]
                train_x_region = train_x[cols]
                test_x_regions = test_x[cols]

                res_curr = res_curr | cca(train_x_region, train_y, test_x_regions, test_x)
                results.append(pd.DataFrame(res_curr, index=[f"region_{region}"]))

            # For DTI features, iterate through the 50 parcels in the white matter atlas
            for region in range(50):
                res_curr = {
                    "Dataset": dataset, "Repeat": np.floor(fold / n_folds),
                    "Fold": fold % n_folds, "Target": target}

                cols = [f"{feature}_{region}" for feature in features_d]
                train_x_d = train_x[cols]
                test_x_d = test_x[cols]

                res_curr = res_curr | cca(train_x_d, train_y, test_x_d, test_x)
                results.append(pd.DataFrame(res_curr, index=[f"dti_region_{region}"]))

# Save all resutls together
pd.concat(results, axis="index").to_csv(Path(args.out_dir, "fusion_cca_results.csv"))
