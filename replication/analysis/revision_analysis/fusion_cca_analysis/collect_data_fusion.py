from pathlib import Path
import argparse

import datalad.api as dl
import numpy as np
import pandas as pd


task_list = {
    "HCP-YA": [
        "tfMRI_EMOTION", "tfMRI_GAMBLING", "tfMRI_LANGUAGE", "tfMRI_MOTOR", "tfMRI_WM",
        "tfMRI_RELATIONAL", "tfMRI_SOCIAL"],
    "HCP-A": ["tfMRI_CARIT_PA", "tfMRI_FACENAME_PA", "tfMRI_VISMOTOR_PA"],
    "HCP-D": ["tfMRI_CARIT", "tfMRI_EMOTION", "tfMRI_GUESSING"]}


def read_features(data_file: Path, key: str) -> pd.DataFrame:
    if key == "rs_stats":
        data = pd.DataFrame(pd.read_hdf(data_file, "rs_par_level3"))
    elif key in ["d_fa", "d_md", "d_ad", "d_rd"]:
        key_name = key.split("d_")[1]
        data = pd.DataFrame(pd.read_hdf(data_file, f"{key_name}_{subject}")).T
    else:
        data = pd.DataFrame(pd.read_hdf(data_file, f"{key}_level3"))
    data = data.replace(-np.inf, 0)
    data = data.fillna(value=0)
    data.columns = [f"{key}_{col}" for col in range(len(data.columns))]
    data.index = [subject]
    return data


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
sublist = pd.read_table(
    Path(args.sublist_dir, f"{args.dataset}_allRun.csv"), header=None, dtype=str).squeeze("columns")

# DTI feature file
dti_file = Path(root_mfe_dir, f"{args.dataset}_dti.h5")
dl.get(dti_file, dataset=root_mfe_dir)

# Feature types
features_conn = (
    ["rs_sfc", "rs_dfc", "rs_ec", "d_scc", "d_scl"]
    + [f"{task}_sfc" for task in task_list[args.dataset]]
    + [f"{task}_ec" for task in task_list[args.dataset]])
features_arr = ["rs_par", "s_myelin", "s_gmv", "s_cs", "s_ct"]
features_d = ["d_fa", "d_md", "d_ad", "d_rd"]

# Output file paths
y_out = Path(args.out_dir, f"{args.dataset}_y.csv")
conf_out = Path(args.out_dir, f"{args.dataset}_conf.csv")
x_conn_out = Path(args.out_dir, f"{args.dataset}_x_conn.csv")
x_arr_out = Path(args.out_dir, f"{args.dataset}_x_arr.csv")
x_d_out = Path(args.out_dir, f"{args.dataset}_x_d.csv")

# Iterate through subjects in the dataset, writing output incrementally to avoid OOM
for i, subject in enumerate(sublist):
    header = bool(not i)
    sub_file = Path(root_mfe_dir, args.dataset, f"{subject}.h5")
    dl.get(sub_file, dataset=root_mfe_dir)

    # Get phenotypes and confounds
    y_df = pd.DataFrame(pd.read_hdf(sub_file, "phenotype"))
    y_df.to_csv(y_out, mode="a", header=header)
    conf_df = pd.DataFrame(pd.read_hdf(sub_file, "confound"))
    conf_df.to_csv(conf_out, mode="a", header=header)

    # Connectivity features
    x_conn = []
    for feature in features_conn:
        x_conn.append(read_features(sub_file, feature))
    x_conn_df = pd.concat(x_conn, axis="columns")
    x_conn_df.to_csv(x_conn_out, mode="a", header=header)

    # Region-wise features
    x_arr = []
    for feature in features_arr:
        x_arr.append(read_features(sub_file, feature))
    x_arr_df = pd.concat(x_arr, axis="columns")
    x_arr_df.to_csv(x_arr_out, mode="a", header=header)

    # DTI features
    x_d = []
    for feature in features_d:
        x_d.append(read_features(dti_file, feature))
    x_d_df = pd.concat(x_d, axis="columns")
    x_d_df.to_csv(x_d_out, mode="a", header=header)

    print(f"Extracted data for {args.dataset} subject {i}: {subject}")
    dl.drop(sub_file, dataset=root_mfe_dir)
dl.drop(dti_file, dataset=root_mfe_dir)

dl.remove(dataset=root_mfe_dir, reckless="kill")
