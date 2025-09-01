from pathlib import Path
import argparse
import itertools

from scipy.stats import t
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd


targets = {
    "Cognition": [
        "totalcogcomp", "crycogcomp", "fluidcogcomp", "cardsort", "flanker", "reading", "picvocab",
        "procspeed", "listsort"],
    "Mental health": [
        "anger", "fear", "sadness", "posaffect", "emotsupp", "friendship", "loneliness"],
    "Personality": ["neoffi_n", "neoffi_e", "neoffi_o", "neoffi_a", "neoffi_c"]}
target_names = {
    "totalcogcomp": "Total cognition", "crycogcomp": "Crystallized cognition",
    "fluidcogcomp": "Fluid cognition", "cardsort": "Cognitive flexibility",
    "flanker": "Inhibitory control", "reading": "Reading", "picvocab": "Picture vocabulary",
    "procspeed": "Processing speed", "listsort": "Working memory", "anger": "Anger affect",
    "fear": "Fear affect", "sadness": "Sadness", "posaffect": "Positive affect",
    "emotsupp": "Emotional Support", "friendship": "Friendship", "loneliness": "Loneliness",
    "neoffi_n": "Neuroticism (NEO)", "neoffi_e": "Extraversion (NEO)", "neoffi_o": "Openness (NEO)",
    "neoffi_a": "Agreeableness (NEO)", "neoffi_c": "Conscientiousness (NEO)"}
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
    "rs_sfc": "rest static FC", "rs_dfc": "time-varying FC", "rs_ec": "rest EC",
    "rs_stats": "rest network statisics", "rs_grad": "rest gradients",
    "tfMRI_CARIT_PA_sfc": "CARIT static FC", "tfMRI_FACENAME_PA_sfc": "FACENAME static FC",
    "tfMRI_VISMOTOR_PA_sfc": "VISMOTOR static FC", "tfMRI_CARIT_PA_ec": "CARIT EC",
    "tfMRI_FACENAME_PA_ec": "FACENAME EC", "tfMRI_VISMOTOR_PA_ec": "VISMOTOR EC",
    "tfMRI_CARIT_sfc": "CARIT static FC", "tfMRI_EMOTION_sfc": "EMOTION static FC",
    "tfMRI_GUESSING_sfc": "GUESSING static FC", "tfMRI_CARIT_ec": "CARIT EC",
    "tfMRI_EMOTION_ec": "EMOTION EC", "tfMRI_GUESSING_ec": "GUESSING EC",
    "tfMRI_GAMBLING_sfc": "GAMBLING static FC", "tfMRI_LANGUAGE_sfc": "LANGUAGE static FC",
    "tfMRI_MOTOR_sfc": "MOTOR static FC", "tfMRI_WM_sfc": "WM static FC",
    "tfMRI_RELATIONAL_sfc": "RELATIONAL static FC", "tfMRI_SOCIAL_sfc": "SOCIAL static FC",
    "tfMRI_GAMBLING_ec": "GAMBLING EC", "tfMRI_LANGUAGE_ec": "LANGUAGE EC",
    "tfMRI_MOTOR_ec": "MOTOR EC", "tfMRI_WM_ec": "WM EC", "tfMRI_RELATIONAL_ec": "RELATIONAL EC",
    "tfMRI_SOCIAL_ec": "SOCIAL EC", "s_myelin": "myelin estimate", "s_gmv": "GMV", "s_cs": "SA",
    "s_ct": "CT", "s_acgmv": "GMV Connectivity", "s_accs": "SA Connectivity",
    "s_acct": "CT Connectivity", "d_scc": "SC (count)", "d_scl": "SC (length)", "d_fa": "FA",
    "d_md": "MD", "d_ad": "AD", "d_rd": "RD", "confounds": "confound", "conf": "confound"}
n_feature_max = {"HCP-A": 25, "HCP-D": 25, "HCP-YA": 33}

parser = argparse.ArgumentParser(
    description="Collect results from all integrated and featurewise predictions",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--datasets", nargs="+", dest="datasets", required=True, help="Datasets")
parser.add_argument(
    "--pred_dir", type=Path, dest="pred_dir", required=True,
    help="absolute path to prediction results")
parser.add_argument(
    "--out_dir", type=Path, dest="out_dir", required=True, help="absolute path to output directory")
parser.add_argument("--n_repeat", type=int, dest="n_repeat", default=10, help="Number of repeats")
parser.add_argument("--n_fold", type=int, dest="n_fold", default=10, help="Number of folds")
parser.add_argument("--level", type=int, dest="level", default=3, help="Granularity level")
parser.add_argument(
    "--overwrite", dest="overwrite", action="store_true", help="overwrite existing output")
args = parser.parse_args()

repeats = range(args.n_repeat)
folds = range(args.n_fold)

# All accuracies
results_file = Path(args.out_dir, "mpp_acc_all.csv")
if (not results_file.exists()) or args.overwrite:
    results = {}
    for dataset in args.datasets:
        for target_domain, target_list in targets.items():
            for target in target_list:
                for repeat, fold in itertools.product(repeats, folds):
                    for acc_type in ["r", "cod"]:
                        for pred_type in ["integrated", "featurewise", "confounds"]:
                            pred_file = Path(args.pred_dir, dataset, f"{pred_type}_{target}.h5")
                            key_cv = f"repeat{repeat}_fold{fold}"
                            results_curr = {
                                    "Dataset": dataset, "Domain": target_domain,
                                    "Target var": target, "Target": target_names[target],
                                    "Repeat": repeat, "Fold": fold, "Accuracy type": acc_type,
                                    "Prediction type": pred_type}

                            if pred_type == "integrated": # Integrated features models
                                for n_feature in range(2, n_feature_max[dataset]+1):
                                    key_if = f"{acc_type}_{pred_type}_{key_cv}_level{args.level}" \
                                             f"_{n_feature}features"
                                    acc = pd.read_hdf(pred_file, key_if).values[0][0]
                                    results[f"{key_if}_{target}_{dataset}"] = results_curr | {
                                        "Number of features": n_feature, "Accuracy": acc,
                                        "Feature": pred_type}

                            elif pred_type == "featurewise": # Featurewise models
                                for feature in features["all"] + features[dataset]:
                                    key_fw = f"{acc_type}_{feature}_{key_cv}_level{args.level}"
                                    acc = pd.read_hdf(pred_file, key_fw).values[0][0]
                                    results[f"{key_fw}_{target}_{dataset}"] = results_curr | {
                                        "Number of features": 1, "Accuracy": acc,
                                        "Feature": feature}

                            elif pred_type == "confounds": # Confounds model
                                key_conf = f"{acc_type}_confound_{key_cv}"
                                acc = pd.read_hdf(pred_file, key_conf).values[0][0]
                                results[f"{key_conf}_{target}_{dataset}"] = results_curr | {
                                    "Number of features": 1, "Accuracy": acc, "Feature": pred_type}

                            print(f"Results table size = {len(results.keys())}")
    results = pd.DataFrame(results).T
    results.to_csv(results_file)
else:
    results = pd.read_csv(results_file)

# Count featurewise models as integrated feature models with 1 feature
results_if = results.loc[results["Prediction type"] == "integrated"]
results_nfeature_file = Path(args.out_dir, "mpp_acc_nfeature.csv")
if (not results_nfeature_file.exists()) or args.overwrite:
    results_nfeature = [results_if]
    for dataset in args.datasets:
        for acc_type in ["r", "cod"]:
            for target_domain, target_list in targets.items():
                for target in target_list:
                    for repeat, fold in itertools.product(repeats, folds):
                        pred_file = Path(args.pred_dir, dataset, f"integrated_{target}.h5")
                        key_cv = f"repeat{repeat}_fold{fold}"
                        ranks = pd.read_hdf(
                            pred_file, f"rank_integrated_{key_cv}_level{args.level}")
                        feature_best = ranks.values[0][0]
                        if feature_best == "conf":
                            feature_best = "confounds"
                        results_fw_curr = results.loc[
                            (results["Dataset"] == dataset)
                            & (results["Accuracy type"] == acc_type)
                            & (results["Target var"] == target)
                            & (results["Repeat"] == repeat) & (results["Fold"] == fold)
                            & (results["Feature"] == feature_best)]
                        results_nfeature.append(results_fw_curr)
    results_nfeature = pd.concat(results_nfeature, axis="index")
    results_nfeature.to_csv(results_nfeature_file)
else:
    results_nfeature = pd.read_csv(results_nfeature_file)

# Feature sets from integrated features models
feature_sets_file = Path(args.out_dir, "mpp_feature_sets.csv")
if (not feature_sets_file.exists()) or args.overwrite:
    results_nfeature.loc[:, "Accuracy"] = results_nfeature["Accuracy"].fillna(0)
    feature_sets = {}
    for dataset in args.datasets:
        for acc_type in ["r", "cod"]:
            for target_domain, target_list in targets.items():
                for target in target_list:
                    feature_set = {
                        "Dataset": dataset, "Target var": target, "Target": target_names[target],
                        "Domain": target_domain, "Accuracy type": acc_type}
                    ind = f"{dataset}_{target}_{acc_type}"

                    acc = np.empty((args.n_repeat, args.n_fold, n_feature_max[dataset]))
                    for n_feature in range(n_feature_max[dataset]):
                        for repeat, fold in itertools.product(repeats, folds):
                            acc_curr = results_nfeature["Accuracy"].loc[
                                (results_nfeature["Dataset"] == dataset)
                                & (results_nfeature["Target var"] == target)
                                & (results_nfeature["Accuracy type"] == acc_type)
                                & (results_nfeature["Number of features"] == n_feature+1)
                                & (results_nfeature["Repeat"] == repeat)
                                & (results_nfeature["Fold"] == fold)]
                            acc[repeat, fold, n_feature] = acc_curr.values[0]

                    # Best feature set
                    best_n = np.argmax(acc.mean(axis=0).mean(axis=0))
                    feature_sets[f"best_{ind}"] = feature_set | {
                        "Type": "best", "Number of features": best_n + 1}

                    # Corrected resampled t test, correcting for multiple comparisons
                    tvals = np.empty((n_feature_max[dataset], n_feature_max[dataset]))
                    pvals = np.empty((n_feature_max[dataset], n_feature_max[dataset]))
                    df = args.n_repeat * args.n_fold - 1
                    for i in range(acc.shape[2]):
                        for j in range(i+1, acc.shape[2]):
                            diff = acc[:, :, i] - acc[:, :, j]
                            tvals[i, j] = diff.mean() / (
                                np.sqrt(1 / (args.n_repeat * args.n_fold)
                                + 1 / (args.n_fold - 1)) * diff.std())
                            pvals[i, j] = 2 * (1 - t.cdf(np.abs(tvals[i, j]), df))
                    pvals_fdr = pvals.copy()
                    pvals_fdr[np.triu_indices_from(pvals, 1)] = multipletests(
                        pvals_fdr[np.triu_indices_from(pvals, 1)], method="fdr_bh")[1]

                    # Necessary feature set
                    nec_n = best_n
                    for n_feature in range(best_n):
                        if pvals_fdr[n_feature, best_n] >= 0.05:
                            nec_n = n_feature
                            break
                    feature_sets[f"nec_{ind}"] = feature_set | {
                        "Type": "necessary", "Number of features": nec_n + 1}
    feature_sets = pd.DataFrame(feature_sets).T
    feature_sets.to_csv(feature_sets_file)
else:
    feature_sets = pd.read_csv(feature_sets_file)

# Necessary feature set: accuracies, features
nec_features_file = Path(args.out_dir, "mpp_nec_features.csv")
nec_acc_file = Path(args.out_dir, "mpp_nec_acc.csv")
if (not nec_features_file.exists() or not nec_acc_file.exists()) or args.overwrite:
    nec_features = {}
    nec_acc = {}
    for dataset in args.datasets:
        for acc_type in ["r", "cod"]:
            for target_domain, target_list in targets.items():
                for target in target_list:
                    nec_n = feature_sets["Number of features"].loc[
                        (feature_sets["Dataset"] == dataset)
                        & (feature_sets["Target var"] == target)
                        & (feature_sets["Accuracy type"] == acc_type)
                        & (feature_sets["Type"] == "necessary")].values[0]
                    ind = f"{dataset}_{target}_{acc_type}"
                    pred_file =  Path(args.pred_dir, dataset, f"integrated_{target}.h5")
                    nec = {
                        "Dataset": dataset, "Accuracy type": acc_type, "Target var": target,
                        "Target": target_names[target], "Domain": target_domain}

                    for repeat, fold in itertools.product(repeats, folds):
                        key = f"repeat{repeat}_fold{fold}"
                        if nec_n == 1:
                            results_feature = results_nfeature.loc[
                                (results_nfeature["Dataset"] == dataset)
                                & (results_nfeature["Accuracy type"] == acc_type)
                                & (results_nfeature["Target var"] == target)
                                & (results_nfeature["Number of features"] == 1)]
                            feature = results_feature["Feature"].iloc[0]
                            nec_features[f"{ind}_{key}_0"] = nec | {
                                "Feature": feature_names[feature]}
                            acc = results_feature["Accuracy"].loc[
                                (results_feature["Repeat"] == repeat)
                                & (results_feature["Fold"] == fold)].values[0]
                            nec_acc[f"{ind}_{key}"] = nec | {"Accuracy": acc}
                        else:
                            ranks = pd.read_hdf(
                                pred_file, f"rank_integrated_{key}_level{args.level}")
                            for i in range(nec_n):
                                feature = ranks.values[i][0]
                                nec_features[f"{ind}_{key}_{i}"] = nec | {
                                    "Feature": feature_names[feature]}
                            key_acc = f"{acc_type}_integrated_{key}_level{args.level}" \
                                      f"_{nec_n}features"
                            acc = pd.read_hdf(pred_file, key_acc).values[0][0]
                            nec_acc[f"{ind}_{key}"] = nec | {"Accuracy": acc}
    nec_features = pd.DataFrame(nec_features).T
    nec_features.to_csv(nec_features_file)
    nec_acc = pd.DataFrame(nec_acc).T
    nec_acc.to_csv(nec_acc_file)
else:
    nec_features = pd.read_csv(nec_features_file)
    nec_acc = pd.read_csv(nec_acc_file)

# Necessary feature set: only targets with viable prediction (COD R2 > 0)
for dataset in args.datasets:
    nec_acc_curr = nec_acc.loc[
        (nec_acc["Accuracy type"] == "cod") & (nec_acc["Dataset"] == dataset)]
    nec_acc_avg = nec_acc_curr.groupby("Target")["Accuracy"].mean()
    targets_dataset = nec_acc_avg.loc[nec_acc_avg > 0].index.tolist()
    if targets_dataset:
        nec_freq_file = Path(args.out_dir, f"mpp_nec_freq_{dataset}.csv")
        if (not nec_freq_file.exists()) or args.overwrite:
            nec_freq = {}
            for target in targets_dataset:
                target_pd = nec_features.loc[
                    (nec_features["Dataset"] == dataset) & (nec_features["Target"] == target)]
                nec_freq[target] = {}
                for feature in features["all"]+features[dataset]+["conf"]:
                    feature_name = feature_names[feature]
                    feature_pd = target_pd.loc[target_pd["Feature"] == feature_name]
                    nec_freq[target][feature_name] = feature_pd.shape[0] / target_pd.shape[0]
            nec_freq = pd.DataFrame(nec_freq)
            nec_freq.to_csv(nec_freq_file)
