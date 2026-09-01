from pathlib import Path
import argparse

import datalad.api as dl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser(
    description="Plot SA variability distribution",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument(
    "--sublist_dir", type=Path, dest="sublist_dir", required=True,
    help="Absolute path to sublsit directory")
parser.add_argument(
    "--data_out_dir", type=Path, dest="data_out_dir", required=True,
    help="Absolute path to output data directory")
parser.add_argument(
    "--plot_out_dir", type=Path, dest="plot_out_dir", required=True,
    help="Absolute path to output image directory")
parser.add_argument(
    "--work_dir", type=Path, dest="work_dir", required=True, help="Absolute path to work directory")
parser.add_argument(
    "--data_file", type=str, dest="data_file", default="", help="Existing SA variance data")
args = parser.parse_args()

sns.set_theme(style="white", context="paper", font_scale=1.5, font="Arial")
datasets = ["HCP-D", "HCP-YA", "HCP-A"]
mfe_url = "git@gin.g-node.org:/jadecci/multimodal_features.git"

# Get SA variability data
if args.data_file:
    data = pd.read_csv(args.data_file, header=0, index_col=[0])
else:
    # Install dataset with multimodal features collected for prediction
    root_mfe_dir = Path(args.work_dir, "mfe_features")
    dl.install(root_mfe_dir, source=mfe_url)

    # Collect SA data for each dataset
    data = dict.fromkeys(datasets)
    for dataset in datasets:
        sublist = pd.read_table(
            Path(args.sublist_dir, f"{dataset}_allRun.csv"),
            header=None, dtype=str).squeeze("columns")

        # Collect all SA data in the dataset
        data_dataset = []
        for subject in sublist:
            sub_file = Path(root_mfe_dir, dataset, f"{subject}.h5")
            dl.get(sub_file, dataset=root_mfe_dir)
            data_dataset.append(pd.read_hdf(sub_file, "s_cs_level3"))
            dl.drop(sub_file, dataset=root_mfe_dir)
        data_dataset = pd.concat(data_dataset, axis="index", join="inner")

        # dataset-wide variance of SA in each brain region
        data[dataset] = data_dataset.var()

    # Write output data
    data = pd.DataFrame(data).T
    data.to_csv(Path(args.data_out_dir, "mfe_sa_var.csv"))
    dl.remove(dataset=root_mfe_dir, reckless="kill")

# Plot SA variability distribution
data_long = pd.wide_to_long(
    data.assign(Dataset=data.index), stubnames="s_cs_", i="Dataset", j="Region")
g = sns.displot(
    kind="hist", data=data_long, x="s_cs_", row="Dataset", stat="percent", kde=True,
    color="dimgray", height=2, aspect=4, bins=100)
g.set_xlabels("Interindividual variability of cortical surface area")

# Add indication of median value
for ax, dataset in zip(g.axes.flatten(), datasets):
    ax.axvline(data.loc[dataset].median(), color="red", linestyle="--")

# Save plot
sa_file = Path(args.plot_out_dir, "sa_var_dis.png")
plt.savefig(sa_file, dpi=500)
plt.close()
