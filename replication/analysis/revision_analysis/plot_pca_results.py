from pathlib import Path
import argparse
import itertools

from baycomp import two_on_single
from scipy.stats import t
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


target = "totalcogcomp"
target_name = "Total cognition"
features = {
    "all": [
        "rs_sfc", "rs_dfc", "rs_ec", "rs_stats", "rs_grad", "s_myelin", "s_gmv", "s_cs", "s_ct",
        "s_acgmv", "s_accs", "s_acct", "d_scc", "d_scl", "d_fa", "d_md", "d_ad", "d_rd"],
    "HCP-A": [
        "tfMRI_CARIT_PA_sfc", "tfMRI_FACENAME_PA_sfc", "tfMRI_VISMOTOR_PA_sfc", "tfMRI_CARIT_PA_ec",
        "tfMRI_FACENAME_PA_ec", "tfMRI_VISMOTOR_PA_ec"],
    "HCP-D": [
        "tfMRI_CARIT_sfc", "tfMRI_EMOTION_sfc", "tfMRI_GUESSING_sfc", "tfMRI_CARIT_ec",
        "tfMRI_EMOTION_ec", "tfMRI_GUESSING_ec"],
    "HCP-YA": [
        "tfMRI_EMOTION_sfc", "tfMRI_GAMBLING_sfc", "tfMRI_LANGUAGE_sfc", "tfMRI_MOTOR_sfc",
        "tfMRI_WM_sfc", "tfMRI_RELATIONAL_sfc", "tfMRI_SOCIAL_sfc", "tfMRI_EMOTION_ec",
        "tfMRI_GAMBLING_ec", "tfMRI_LANGUAGE_ec", "tfMRI_MOTOR_ec", "tfMRI_WM_ec",
        "tfMRI_RELATIONAL_ec", "tfMRI_SOCIAL_ec"]}
feature_names = {
    "rs_sfc": "rest model-free FC", "rs_dfc": "time-varying FC", "rs_ec": "rest EC",
    "rs_stats": "rest network statisics", "rs_grad": "rest gradients",
    "tfMRI_CARIT_PA_sfc": "CARIT model-free FC", "tfMRI_FACENAME_PA_sfc": "FACENAME model-free FC",
    "tfMRI_VISMOTOR_PA_sfc": "VISMOTOR model-free FC", "tfMRI_CARIT_PA_ec": "CARIT EC",
    "tfMRI_FACENAME_PA_ec": "FACENAME EC", "tfMRI_VISMOTOR_PA_ec": "VISMOTOR EC",
    "tfMRI_CARIT_sfc": "CARIT model-free FC", "tfMRI_EMOTION_sfc": "EMOTION model-free FC",
    "tfMRI_GUESSING_sfc": "GUESSING model-free FC", "tfMRI_CARIT_ec": "CARIT EC",
    "tfMRI_EMOTION_ec": "EMOTION EC", "tfMRI_GUESSING_ec": "GUESSING EC",
    "tfMRI_GAMBLING_sfc": "GAMBLING model-free FC", "tfMRI_LANGUAGE_sfc": "LANGUAGE model-free FC",
    "tfMRI_MOTOR_sfc": "MOTOR model-free FC", "tfMRI_WM_sfc": "WM model-free FC",
    "tfMRI_RELATIONAL_sfc": "RELATIONAL model-free FC", "tfMRI_SOCIAL_sfc": "SOCIAL model-free FC",
    "tfMRI_GAMBLING_ec": "GAMBLING EC", "tfMRI_LANGUAGE_ec": "LANGUAGE EC",
    "tfMRI_MOTOR_ec": "MOTOR EC", "tfMRI_WM_ec": "WM EC", "tfMRI_RELATIONAL_ec": "RELATIONAL EC",
    "tfMRI_SOCIAL_ec": "SOCIAL EC", "s_myelin": "myelin estimate", "s_gmv": "GMV", "s_cs": "SA",
    "s_ct": "CT", "s_acgmv": "GMV Connectivity", "s_accs": "SA Connectivity",
    "s_acct": "CT Connectivity", "d_scc": "SC (count)", "d_scl": "SC (length)", "d_fa": "FA",
    "d_md": "MD", "d_ad": "AD", "d_rd": "RD"}
n_feature_max = {"HCP-A": 24, "HCP-D": 24, "HCP-YA": 32}
ropes = {"r": 0.05, "cod": 0.025}

parser = argparse.ArgumentParser(
    description="Collect and plot results from all predictions using pca-en model",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument(
    "--pred_dir", type=Path, dest="pred_dir", required=True,
    help="absolute path to prediction results")
parser.add_argument(
    "--out_dir", type=Path, dest="out_dir", required=True, help="absolute path to output directory")
parser.add_argument(
    "--img_dir", type=Path, dest="img_dir", required=True, help="absolute path to image directory")
parser.add_argument("--n_repeat", type=int, dest="n_repeat", default=10, help="Number of repeats")
parser.add_argument("--n_fold", type=int, dest="n_fold", default=10, help="Number of folds")
parser.add_argument("--level", type=int, dest="level", default=3, help="Granularity level")
args = parser.parse_args()

args.out_dir.mkdir(parents=True, exist_ok=True)
repeats = range(args.n_repeat)
folds = range(args.n_fold)
cmap_heat = sns.light_palette(color="orange", n_colors=20)
dataset_order = ["HCP-D", "HCP-A"]
sns.set_theme(style="white", context="paper", font_scale=2, font="Arial")

# Collect results for each dataset
for dataset in dataset_order:
    # All accuracies
    results_file = Path(args.out_dir, f"mpp_pca_acc_{dataset}.csv")
    results = {}
    for repeat, fold in itertools.product(repeats, folds):
        for acc_type in ["r", "cod"]:
            for pred_type in ["integrated", "featurewise"]:
                pred_file = Path(args.pred_dir, f"{dataset}_pca", f"{pred_type}_{target}.h5")
                key_cv = f"repeat{repeat}_fold{fold}_level{args.level}"
                results_curr = {
                    "Target var": target, "Target": target_name, "Repeat": repeat,
                    "Fold": fold, "Accuracy type": acc_type, "Prediction type": pred_type}

                if pred_type == "integrated": # Integrated features models
                    for n_feature in range(2, n_feature_max[dataset]+1):
                        key_if = f"{acc_type}_{pred_type}_{key_cv}_{n_feature}features"
                        acc = pd.read_hdf(pred_file, key_if).values[0][0]
                        results[f"{key_if}_{target}_{dataset}"] = results_curr | {
                            "Number of features": n_feature, "Accuracy": acc,
                            "Feature": pred_type}

                elif pred_type == "featurewise": # Featurewise models
                    for feature in features["all"] + features[dataset]:
                        key_fw = f"{acc_type}_{feature}_{key_cv}"
                        acc = pd.read_hdf(pred_file, key_fw).values[0][0]
                        results[f"{key_fw}_{target}_{dataset}"] = results_curr | {
                            "Number of features": 1, "Accuracy": acc, "Feature": feature}
    results = pd.DataFrame(results).T
    results.to_csv(results_file)

    # Count featurewise models as integrated feature models with 1 feature
    results_if = results.loc[results["Prediction type"] == "integrated"]
    results_nfeature_file = Path(args.out_dir, f"mpp_pca_acc_nfeature_{dataset}.csv")
    results_nfeature = [results_if]
    for acc_type in ["r", "cod"]:
        for repeat, fold in itertools.product(repeats, folds):
            pred_file = Path(args.pred_dir, f"{dataset}_pca", f"integrated_{target}.h5")
            key_cv = f"repeat{repeat}_fold{fold}_level{args.level}"
            ranks = pd.read_hdf(pred_file, f"rank_integrated_{key_cv}")
            feature_best = ranks.values[0][0]
            results_fw_curr = results.loc[
                (results["Accuracy type"] == acc_type) & (results["Repeat"] == repeat)
                & (results["Fold"] == fold) & (results["Feature"] == feature_best)]
            results_nfeature.append(results_fw_curr)
    results_nfeature = pd.concat(results_nfeature, axis="index")
    results_nfeature.to_csv(results_nfeature_file)

    # Feature sets from integrated features models
    feature_sets_file = Path(args.out_dir, f"mpp_pca_feature_sets_{dataset}.csv")
    results_nfeature.loc[:, "Accuracy"] = results_nfeature["Accuracy"].fillna(0)
    feature_sets = {}
    for acc_type in ["r", "cod"]:
        feature_set = {"Target var": target, "Target": target_name, "Accuracy type": acc_type}
        ind = f"{dataset}_{target}_{acc_type}"

        acc = np.empty((args.n_repeat, args.n_fold, n_feature_max[dataset]))
        for n_feature in range(n_feature_max[dataset]):
            for repeat, fold in itertools.product(repeats, folds):
                acc_curr = results_nfeature["Accuracy"].loc[
                    (results_nfeature["Accuracy type"] == acc_type)
                    & (results_nfeature["Number of features"] == n_feature+1)
                    & (results_nfeature["Repeat"] == repeat)
                    & (results_nfeature["Fold"] == fold)]
                acc[repeat, fold, n_feature] = acc_curr.values[0]

        # Best feature set
        acc_mean = acc.mean(axis=0).mean(axis=0)
        best_ind = np.argsort(-acc_mean)[:3] # top 3 best models
        for i in range(3):
            feature_sets[f"best_top{i+1}_{ind}"] = feature_set | {
                "Type": "best", "Number of features": best_ind[i] + 1}

        # Bayesian correlated t-test to determine necessary feature set
        nec_n = best_ind[0]
        for n_feature in range(n_feature_max[dataset]):
            curr_acc = acc[:, :, n_feature].flatten()
            rope_check = 0
            for i in best_ind:
                best_acc = acc[:, :, i].flatten()
                _, p_rope, _ = two_on_single(
                    curr_acc, best_acc, rope=ropes[acc_type], runs=args.n_repeat)
                if p_rope < 0.95:
                    break
                else:
                    rope_check += 1
            if rope_check == 3:
                nec_n = n_feature
                break
        feature_sets[f"nec_{ind}"] = feature_set | {
            "Type": "necessary", "Number of features": nec_n + 1}
    feature_sets = pd.DataFrame(feature_sets).T
    feature_sets.to_csv(feature_sets_file)

    # Necessary feature set: accuracies, features
    nec_features_file = Path(args.out_dir, f"mpp_pca_nec_features_{dataset}.csv")
    nec_acc_file = Path(args.out_dir, f"mpp_pca_nec_acc_{dataset}.csv")
    nec_features = {}
    nec_acc = {}
    for acc_type in ["r", "cod"]:
        nec_n = feature_sets["Number of features"].loc[
            (feature_sets["Accuracy type"] == acc_type)
            & (feature_sets["Type"] == "necessary")].values[0]
        ind = f"{dataset}_{target}_{acc_type}"
        pred_file = Path(args.pred_dir, f"{dataset}_pca", f"integrated_{target}.h5")
        nec = {"Accuracy type": acc_type, "Target var": target, "Target": target_name}

        for repeat, fold in itertools.product(repeats, folds):
            key = f"repeat{repeat}_fold{fold}_level{args.level}"
            if nec_n == 1:
                results_feature = results_nfeature.loc[
                    (results_nfeature["Accuracy type"] == acc_type)
                    & (results_nfeature["Number of features"] == 1)]
                feature = results_feature["Feature"].iloc[0]
                nec_features[f"{ind}_{key}_0"] = nec | {"Feature": feature_names[feature]}
                acc = results_feature["Accuracy"].loc[
                    (results_feature["Repeat"] == repeat)
                    & (results_feature["Fold"] == fold)].values[0]
                nec_acc[f"{ind}_{key}"] = nec | {"Accuracy": acc}
            else:
                ranks = pd.read_hdf(pred_file, f"rank_integrated_{key}")
                for i in range(nec_n):
                    feature = ranks.values[i][0]
                    nec_features[f"{ind}_{key}_{i}"] = nec | {"Feature": feature_names[feature]}
                key_acc = f"{acc_type}_integrated_{key}_{nec_n}features"
                acc = pd.read_hdf(pred_file, key_acc).values[0][0]
                nec_acc[f"{ind}_{key}"] = nec | {"Accuracy": acc}
    nec_features = pd.DataFrame(nec_features).T
    nec_features.to_csv(nec_features_file)
    nec_acc = pd.DataFrame(nec_acc).T
    nec_acc.to_csv(nec_acc_file)

    # Necessary feature set: only targets with viable prediction (COD R2 > 0)
    nec_acc_curr = nec_acc.loc[nec_acc["Accuracy type"] == "cod"]
    nec_freq_file = Path(args.out_dir, f"mpp_pca_nec_freq_{dataset}.csv")
    nec_freq = {}
    df = nec_acc_curr.shape[0]
    scale = nec_acc_curr["Accuracy"].std() / np.sqrt(df)
    nec_acc_ci = t.interval(0.95, df=(df - 1), loc=nec_acc_curr["Accuracy"].mean(), scale=scale)

    if nec_acc_ci[0] > 0 and nec_acc_curr["Accuracy"].mean() >= 0.05:
        feature_pd = feature_sets.loc[
            (feature_sets["Accuracy type"] == "cod") & (feature_sets["Type"] == "necessary")]
        print(f"Valid model for {dataset} ({feature_pd['Number of features'].values[0]} features)")

        nec_freq[target] = {}
        for feature in features["all"]+features[dataset]:
            feature_name = feature_names[feature]
            feature_pd = nec_features.loc[nec_features["Feature"] == feature_name]
            nec_freq[target][feature_name] = feature_pd.shape[0] / nec_features.shape[0]
    nec_freq = pd.DataFrame(nec_freq)
    nec_freq.to_csv(nec_freq_file)

# Plot results
# Integrated features models: accuracy vs. number of features
for acc_type in ["r", "cod"]:
    if_acc_file = Path(args.img_dir, f"pca_if_acc_{acc_type}.png")
    results = []
    for ds in dataset_order:
        res_pca = pd.read_csv(Path(args.out_dir, f"mpp_pca_acc_nfeature_{ds}.csv"), index_col=0)
        res_pca = res_pca.assign(Dataset=ds).assign(Model="fusion-pca")
        res_pca["Dataset - target"] = res_pca["Dataset"] + " - " + res_pca["Target"]
        results.append(res_pca)

        res_def = pd.read_csv(Path(args.out_dir, f"mpp_acc_nfeature_{ds}.csv"), index_col=0)
        res_def = res_def.loc[res_def["Target var"] == target]
        res_def = res_def.assign(Dataset=ds).assign(Model="default")
        res_def["Dataset - target"] = res_def["Dataset"] + " - " + res_def["Target"]
        results.append(res_def)
    results = pd.concat(results, axis="index").reset_index(drop=True)

    g = sns.relplot(
        data=results.loc[results["Accuracy type"] == acc_type], kind="line",
        x="Number of features", y="Accuracy", hue="Model", col="Dataset - target",
        style="Model", palette="Set2", markers=True, ms=10, errorbar=("ci", 95),
        height=15, aspect=0.4, facet_kws={"sharey": True, "sharex": False})
    for ax in g.axes.flat:
        ax.axhline(color="black", linestyle="--")
    plt.savefig(if_acc_file, bbox_inches="tight", dpi=500)
    plt.close()

# Necessary feature set: necessary feature frequency by target
for dataset in dataset_order:
    nec_freq_file = Path(args.img_dir, f"pca_nec_freq_{dataset}.png")
    nec_freq = pd.read_csv(Path(args.out_dir, f"mpp_pca_nec_freq_{dataset}.csv"), index_col=0)
    annot = nec_freq.map("{:.0%}".format).astype(str).replace("0%", "")
    nec_freq = nec_freq.mul(100)
    plt.figure(figsize=(nec_freq.shape[1]*1.2, nec_freq.shape[0]*0.5))
    ax = sns.heatmap(
        data=nec_freq, vmin=0, vmax=100, cbar=True, cmap=cmap_heat, annot=annot, fmt="",
        linewidth=.5, cbar_kws={"format": "%d%%", "label": "Probability"})
    ax.tick_params(axis="x", labelrotation=45, labeltop=True, labelbottom=False)
    ax.set_xticklabels([target_name], ha="left")
    plt.savefig(nec_freq_file, bbox_inches="tight", dpi=500)
    plt.close()
