from pathlib import Path
from timeit import default_timer
import argparse

from statsmodels.formula.api import ols
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description="Compute cross-validation-specific features for fusion CCA models",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--dataset", type=str, dest="dataset", required=True, help="Dataset")
parser.add_argument("--fold", type=int, required=True, help="Fold number (0 to 99)")
parser.add_argument(
    "--data_dir", type=Path, dest="data_dir", required=True, help="Absolute path to collected data")
parser.add_argument(
    "--out_dir", type=Path, dest="out_dir", required=True, help="Absolute path to output directory")
args = parser.parse_args()

# Set-up
args.out_dir.mkdir(parents=True, exist_ok=True)
n_repeats = 10
n_folds = 10

# Collected data for all subjects
x = pd.read_csv(Path(args.data_dir, f"{args.dataset}_x_arr.csv"), header=0, index_col=[0])
subjects = x.index.to_list()

# CV indices
train_file = Path(args.data_dir, f"{args.dataset}_train_fold{args.fold}.csv")
train_ind = pd.read_csv(train_file, header=None).squeeze().to_list()
train_x = x.iloc[train_ind]

# Compute structural co-registration features
for feature in ["s_gmv", "s_cs", "s_ct"]:
    time_0 = default_timer()
    print(f"Computing {feature} features for fold {args.fold}")
    nparc = 350 if feature == "s_gmv" else 300
    cols = [f"{feature}_{region}" for region in range(nparc)]
    ac_name = f"s_ac{feature.split('s_')[1]}"

    # Get features from training set
    features = train_x[cols].apply(pd.to_numeric)
    features.columns = range(nparc)
    features = features.join(pd.DataFrame({"mean": features.mean(axis=1)}))

    # Estimate paramaters
    params = {}
    for i in range(nparc):
        for j in range(nparc):
            res = ols(f"features[{i}] ~ features[{j}] + mean", data=features).fit()
            params[f"{i}_{j}"] = [
                res.params["Intercept"], res.params[f"features[{j}]"], res.params["mean"]]
    
    # Apply to all subjects
    ac_feature = []
    for subject in subjects:
        features_sub = x.loc[subject, cols]
        ac_curr = []
        for i in range(nparc):
            for j in range(nparc):
                param_curr = params[f"{i}_{j}"]
                morph_curr = features_sub[f"{feature}_{i}"]
                ac_curr.append((
                    param_curr[0] + param_curr[1] * morph_curr + param_curr[2]
                    * features_sub.mean()))
        ac_curr = pd.DataFrame(ac_curr).T
        ac_curr.columns = [
            f"{ac_name}_{col}" for col in range(len(ac_curr.columns))]
        ac_curr.index = [subject]
        ac_feature.append(ac_curr)
    ac_feature_df = pd.concat(ac_feature, axis="index")

    ac_out = Path(args.out_dir, f"{args.dataset}_{ac_name}_fold{args.fold}.csv")
    ac_feature_df.to_csv(ac_out)
    time_1 = default_timer()
    print(f"Finished computing {feature} features for fold {args.fold} in {time_1 - time_0:.2f}s")
