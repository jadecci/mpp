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
    description="Plot correlation between psychometric variables",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument(
    "--data_dir", type=Path, dest="data_dir", required=True, help="absolute path to data directory")
parser.add_argument(
    "--sublist_dir", type=Path, dest="sublist_dir", required=True,
    help="absolute path to subject list directory")
parser.add_argument(
    "--out_dir", type=Path, dest="out_dir", required=True, help="absolute path to output directory")
parser.add_argument(
    "--overwrite", dest="overwrite", action="store_true", help="overwrite existing output")
args = parser.parse_args()

dataset_order = ["HCP-D", "HCP-YA", "HCP-A"]
sns.set_theme(style="white", context="paper", font_scale=1.5, font="Arial")

# Collect phenotype data
data = []
for dataset in dataset_order:
    data_curr = pd.read_csv(Path(args.data_dir, f"{dataset}_FD_phenotype.csv"), index_col=0)
    data.append(data_curr.replace(999, np.nan))
data = pd.concat(data, axis="index", join="inner")

# Compute correlation, averaged across datasets
corr = pd.DataFrame(0, columns=phenos, index=phenos)
for dataset in dataset_order:
    data_curr = data[phenos].loc[data["Dataset"] == dataset]
    corr_curr = data_curr.corr(numeric_only=True)
    corr = corr + corr_curr
corr = corr / len(dataset_order)

# Plot correlations
corr_file = Path(args.out_dir, "pheno_corr.png")
if (not corr_file.exists()) or args.overwrite:
    fig = plt.figure(figsize=(20, 15), constrained_layout=True)
    gs = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[1, 0.02])
    ax_plot = fig.add_subplot(gs[0])
    ax_cbar = fig.add_subplot(gs[1])
    sns.heatmap(
        data=corr, vmin=-1, vmax=1, center=0, cmap="vlag", annot=True, fmt=".2f", square=True,
        linewidths=0.5, ax=ax_plot, cbar_ax=ax_cbar, cbar_kws={"orientation": "horizontal"})
    ax_plot.set_xticklabels(labels, ha="left")
    ax_plot.set_yticklabels(labels, ha="right")
    ax_plot.tick_params(axis="x", labelrotation=45, labeltop=True, labelbottom=False)
    plt.savefig(corr_file, bbox_inches="tight", dpi=500)
    plt.close()
