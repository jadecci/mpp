from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


features = [
    "rest model-free FC", "time-varying FC", "rest EC", "rest network statisics", "rest gradients",
    "CARIT model-free FC", "FACENAME model-free FC", "VISMOTOR model-free FC", "CARIT EC",
    "FACENAME EC", "VISMOTOR EC", "CARIT model-free FC", "EMOTION model-free FC",
    "GUESSING model-free FC", "CARIT EC", "EMOTION EC", "GUESSING EC", "GAMBLING model-free FC",
    "LANGUAGE model-free FC", "MOTOR model-free FC", "WM model-free FC", "RELATIONAL model-free FC",
    "SOCIAL model-free FC", "GAMBLING EC", "LANGUAGE EC", "MOTOR EC", "WM EC", "RELATIONAL EC",
    "SOCIAL EC", "myelin estimate", "GMV", "SA", "CT", "GMV Connectivity", "SA Connectivity",
    "CT Connectivity", "SC (count)", "SC (length)", "FA", "MD", "AD", "RD", "confound"]
targets = [
    "Total cognition", "Crystallized cognition", "Fluid cognition", "Cognitive flexibility",
    "Inhibitory control", "Reading", "Picture vocabulary", "Processing speed", "Working memory",
    "Anger affect", "Fear affect", "Sadness", "Positive affect", "Emotional Support", "Friendship",
    "Loneliness", "Neuroticism (NEO)", "Extraversion (NEO)", "Openness (NEO)",
    "Agreeableness (NEO)", "Conscientiousness (NEO)"]

parser = argparse.ArgumentParser(
    description="Plot figures using collected results",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument(
    "--res_dir", type=Path, dest="res_dir", required=True,
    help="absolute path to collected prediction results")
parser.add_argument(
    "--out_dir", type=Path, dest="out_dir", required=True, help="absolute path to output directory")
parser.add_argument(
    "--overwrite", dest="overwrite", action="store_true", help="overwrite existing output")
args = parser.parse_args()

cmap = (
        sns.color_palette(palette="Oranges_r", n_colors=9)
        + sns.color_palette(palette="Blues_r", n_colors=7)
        + sns.color_palette(palette="Greens_r", n_colors=5))
cmap_heat = sns.light_palette(color="orange", n_colors=20)
dataset_order = ["HCP-D", "HCP-YA", "HCP-A"]
sns.set_theme(style="white", context="paper", font_scale=2, font="Arial")

# Integrated features models: accuracy vs. number of features
for acc_type in ["r", "cod"]:
    if_acc_file = Path(args.out_dir, f"if_acc_{acc_type}.png")
    if (not if_acc_file.exists()) or args.overwrite:
        results = pd.read_csv(Path(args.res_dir, "mpp_acc_nfeature.csv"))
        g = sns.relplot(
            data=results.loc[results["Accuracy type"] == acc_type], kind="line",
            x="Number of features", y="Accuracy", hue="Target", col="Dataset",
            col_order=dataset_order, palette=cmap, marker="o", errorbar=("ci", 95),
            height=15, aspect=0.4, facet_kws={"sharey": True, "sharex": False})
        for ax in g.axes.flat:
            ax.axhline(color="black", linestyle="--")
        plt.savefig(if_acc_file, bbox_inches="tight", dpi=500)
        plt.close()

# Integrated features models: histogram of number of features
for acc_type in ["r", "cod"]:
    if_nfeature_file = Path(args.out_dir, f"if_nfeature_{acc_type}.png")
    if (not if_nfeature_file.exists()) or args.overwrite:
        feature_sets = pd.read_csv(Path(args.res_dir, "mpp_feature_sets.csv"))
        sns.displot(
            data=feature_sets.loc[feature_sets["Accuracy type"] == acc_type], kind="hist",
            x="Number of features", col="Type", hue="Dataset", palette="Greys", binwidth=1,
            element="step", multiple="stack", height=4, aspect=1.5,
            facet_kws={"legend_out": True, "sharex": True})
        plt.savefig(if_nfeature_file, bbox_inches="tight", dpi=500)
        plt.close()

# Necessary feature set: accuracies
for acc_type in ["r", "cod"]:
    nec_acc_file = Path(args.out_dir, f"nec_acc_{acc_type}.png")
    if (not nec_acc_file.exists()) or args.overwrite:
        nec_acc = pd.read_csv(Path(args.res_dir, "mpp_nec_acc.csv"))
        g = sns.catplot(
            data=nec_acc.loc[nec_acc["Accuracy type"] == acc_type], kind="box", y="Target",
            x="Accuracy", hue="Target", col="Dataset", col_order=dataset_order, orient="h",
            palette=cmap, height=10, aspect=0.5, showfliers=False, showcaps=False,
            boxprops={"linewidth": 0}, whiskerprops={"color": "lightgray", "linewidth": 1.5},
            medianprops={"linewidth": 2.5}, legend=False, sharey=True, sharex=True)
        for ax in g.axes.flat:
            ax.axvline(color="lightgray", linestyle="--")
        plt.savefig(nec_acc_file, bbox_inches="tight", dpi=500)
        plt.close()

# Necessary feature set: necessary feature frequency by dataset
nec_features_file = Path(args.out_dir, f"nec_features_dataset.png")
if (not nec_features_file.exists()) or args.overwrite:
    nec_features = pd.read_csv(Path(args.res_dir, "mpp_nec_features.csv"))
    nec_sorted = nec_features.sort_values(
        by="Feature", key=lambda column: column.map(lambda x: features.index(x)))
    sns.displot(
        data=nec_sorted, kind="hist", y="Feature", col="Dataset", col_order=dataset_order,
        hue="Target", hue_order=targets, stat="percent", palette=cmap, multiple="stack",
        height=15, aspect=0.4, facet_kws={"legend_out": True, "sharey": True})
    plt.savefig(nec_features_file, bbox_inches="tight", dpi=500)
    plt.close()

# Necessary feature set: necessary feature frequency by target
for dataset in ["HCP-A", "HCP-YA", "HCP-D"]:
    nec_freq_file = Path(args.out_dir, f"nec_freq_{dataset}.png")
    if (not nec_freq_file.exists()) or args.overwrite:
        nec_freq = pd.read_csv(Path(args.res_dir, f"mpp_nec_freq_{dataset}.csv"), index_col=0)
        annot = nec_freq.map("{:.0%}".format).astype(str).replace("0%", "")
        nec_freq = nec_freq.mul(100)
        plt.figure(figsize=(nec_freq.shape[1]*1.2, nec_freq.shape[0]*0.5))
        ax = sns.heatmap(
            data=nec_freq, vmin=0, cbar=True, cmap=cmap_heat, annot=annot, fmt="", linewidth=.5,
            cbar_kws={"format": "%d%%", "label": "Probability"})
        ax.tick_params(axis="x", labelrotation=45, labeltop=True, labelbottom=False)
        ax.set_xticklabels(ax.get_xticklabels(), ha="left")
        plt.savefig(nec_freq_file, bbox_inches="tight", dpi=500)
        plt.close()
