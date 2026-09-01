from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

phenos = [
    "totalcogcomp", "fluidcogcomp", "crycogcomp", "cardsort", "flanker", "reading", "picvocab",
    "procspeed", "listsort", "anger", "fear", "sadness", "posaffect", "emotsupp", "friendship",
    "loneliness", "neoffi_n", "neoffi_e", "neoffi_o", "neoffi_a", "neoffi_c"]
labels = [
    "Total cognition", "Fluid cognition", "Crystallized cognition", "Cognitive flexibility",
    "Inhibitory control", "Reading", "Picture vocabulary", "Processing speed", "Working memory",
    "Anger affect", "Fear affect", "Sadness", "Positive effect", "Emotional support", "Friendship",
    "Loneliness", "Neuroticism (NEO)", "Extraversion (NEO)", "Openness (NEO)",
    "Agreeableness (NEO)", "Conscientiousness (NEO)"]

parser = argparse.ArgumentParser(
    description="Plot FD distribution and correlation with pyschometric variables",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument(
    "--data_dir", type=Path, dest="data_dir", required=True,
    help="absolute path to FD data")
parser.add_argument(
    "--out_dir", type=Path, dest="out_dir", required=True, help="absolute path to output directory")
parser.add_argument(
    "--overwrite", dest="overwrite", action="store_true", help="overwrite existing output")
args = parser.parse_args()

sns.set_theme(style="white", context="paper", font_scale=1.5, font="Arial")

# Collect FD data
datasets = ["HCP-A", "HCP-YA", "HCP-D"]
data = []
for dataset in datasets:
    data_curr = pd.read_csv(Path(args.data_dir, f"{dataset}_FD_phenotype.csv"), index_col=0)
    data.append(data_curr.replace(999, np.nan))
data = pd.concat(data, axis="index", join="inner")

# Plot FD distribution
fd_dis_file = Path(args.out_dir, "fd_dis.png")
if (not fd_dis_file.exists()) or args.overwrite:
    sns.displot(kind="hist", data=data, x="Mean FD", stat="percent", col="Dataset", color="dimgray")
    plt.savefig(fd_dis_file, dpi=500)
    plt.close()

# Correlation with psychometric variables
corr = []
for dataset in datasets:
    corr_curr = data.loc[data["Dataset"] == dataset].corr(numeric_only=True).loc[["Mean FD"]]
    corr_curr = corr_curr.drop(columns="Mean FD")
    corr_curr = corr_curr.rename(index={"Mean FD": f"{dataset} mean FD"})
    corr.append(corr_curr)
corr = pd.concat(corr, axis="index")

# Plot correlation
fd_corr_file = Path(args.out_dir, "fd_pheno_corr.png")
if (not fd_corr_file.exists()) or args.overwrite:
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    gs = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[1, 0.05])
    ax_plot = fig.add_subplot(gs[0])
    sns.heatmap(
        data=corr[phenos], vmin=-1, vmax=1, center=0, cmap="vlag", annot=True, fmt=".2f",
        square=True, ax=ax_plot, cbar_ax=fig.add_subplot(gs[1]), linewidths=0.5, xticklabels=labels,
        cbar_kws={"orientation": "horizontal"})
    ax_plot.set_xticklabels(labels, ha="left")
    ax_plot.tick_params(axis="x", labelrotation=45, labeltop=True, labelbottom=False)
    plt.savefig(fd_corr_file, dpi=500)
    plt.close()
