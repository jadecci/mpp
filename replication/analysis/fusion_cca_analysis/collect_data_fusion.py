from pathlib import Path
import argparse

import datalad.api as dl
import numpy as np
import pandas as pd

base_feature_list = [
    "rs_sfc", "rs_dfc", "rs_ec", "rs_stats", "s_myelin", "s_gmv", "s_cs", "s_ct", "d_scc", "d_scl",
    "d_fa", "d_md", "d_ad", "d_rd"]
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

parser = argparse.ArgumentParser(
    description="Collect data for fusion CCA analysis",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("--dataset", type=str, dest="dataset", required=True, help="Dataset")
parser.add_argument(
    "--sublist_dir", type=Path, dest="sublist_dir", required=True, help="Sublist directory")
parser.add_argument(
    "--out_dir", type=Path, dest="out_dir", required=True, help="Absolute path to output directory")
parser.add_argument(
    "--work_dir", type=Path, dest="work_dir", required=True, help="Absolute path to work directory")
args = parser.parse_args()

# Install dataset with multimodal features collected for prediction
mfe_url = "git@gin.g-node.org:/jadecci/multimodal_features.git"
root_mfe_dir = Path(args.work_dir, f"{args.dataset}_mfe_features")
dl.install(root_mfe_dir, source=mfe_url)

# Set-up
args.out_dir.mkdir(parents=True, exist_ok=True)
level = 3
features = base_feature_list + task_feature_list[args.dataset]
sublist = pd.read_table(
    Path(args.sublist_dir, f"{args.dataset}_allRun.csv"), header=None, dtype=str).squeeze("columns")

# DTI feature file
dti_file = Path(root_mfe_dir, f"{args.dataset}_dti.h5")
dl.get(dti_file, dataset=root_mfe_dir)

# Output file paths
y_out = Path(args.out_dir, f"{args.dataset}_y.csv")
conf_out = Path(args.out_dir, f"{args.dataset}_conf.csv")
x_out = Path(args.out_dir, f"{args.dataset}_x.csv")

# Iterate through subjects in the dataset, writing output incrementally to avoid OOM
for i, subject in enumerate(sublist):
    sub_file = Path(root_mfe_dir, args.dataset, f"{subject}.h5")
    dl.get(sub_file, dataset=root_mfe_dir)

    # Get phenotypes and confounds
    y_df = pd.DataFrame(pd.read_hdf(sub_file, "phenotype"))
    conf_df = pd.DataFrame(pd.read_hdf(sub_file, "confound"))

    # Iterate through feature types
    x_sub = []
    for feature in features:
        if feature == "rs_stats":
            x_curr = pd.DataFrame(pd.read_hdf(sub_file, "rs_par_level3"))
        elif feature in ["d_fa", "d_md", "d_ad", "d_rd"]:
            feature_name = feature.split("d_")[1]
            x_curr = pd.DataFrame(pd.read_hdf(dti_file, f"{feature_name}_{subject}")).T
        else:
            x_curr = pd.DataFrame(pd.read_hdf(sub_file, f"{feature}_level3"))
        x_curr = x_curr.replace(-np.inf, 0)
        x_curr = x_curr.fillna(value=0)
        x_curr.columns = [f"{feature}_{col}" for col in range(len(x_curr.columns))]
        x_sub.append(x_curr)
    x_df = pd.concat(x_sub, axis="columns")

    # Write this subject's data immediately; only write the header on the first subject
    write_header = i == 0
    y_df.to_csv(y_out, mode="a", header=write_header)
    conf_df.to_csv(conf_out, mode="a", header=write_header)
    x_df.to_csv(x_out, mode="a", header=write_header)

    print(f"Extracted data for {args.dataset} subject {i}: {subject}")
    dl.drop(sub_file, dataset=root_mfe_dir)
dl.drop(dti_file, dataset=root_mfe_dir)

dl.remove(dataset=root_mfe_dir, reckless="kill")
