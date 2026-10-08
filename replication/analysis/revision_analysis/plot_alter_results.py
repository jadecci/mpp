from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


valid_targets = {
    "HCP-D": ["totalcogcomp"],
    "HCP-YA": ["totalcogcomp", "crycogcomp", "fluidcogcomp", "reading", "picvocab", "listsort"],
    "HCP-A": ["totalcogcomp", "crycogcomp", "fluidcogcomp", "reading", "picvocab"]}
target_names = {
    "totalcogcomp": "Total cognition", "crycogcomp": "Crystallized cognition",
    "fluidcogcomp": "Fluid cognition", "reading": "Reading", "picvocab": "Picture vocabulary",
    "listsort": "Working memory"}


parser = argparse.ArgumentParser(
    description="Plot figures for alternative model using collected results",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument(
    "--res_dir", type=Path, dest="res_dir", required=True,
    help="absolute path to collected prediction results")
parser.add_argument(
    "--out_dir", type=Path, dest="out_dir", required=True, help="absolute path to output directory")
parser.add_argument(
    "--overwrite", dest="overwrite", action="store_true", help="overwrite existing output")
args = parser.parse_args()

cmap_heat = sns.light_palette(color="orange", n_colors=20)
dataset_order = ["HCP-D", "HCP-A"]
sns.set_theme(style="white", context="paper", font_scale=2, font="Arial")

# Integrated features models: accuracy vs. number of features
for acc_type in ["r", "cod"]:
    if_acc_file = Path(args.out_dir, f"alter_if_acc_{acc_type}.png")
    if (not if_acc_file.exists()) or args.overwrite:
        results = []
        for ds in ["HCP-D", "HCP-A"]:
            res_alter = pd.read_csv(
                Path(args.res_dir, f"mpp_alter_acc_nfeature_{ds}.csv"), index_col=0)
            res_alter = res_alter.assign(Dataset=ds).assign(Model="alternative")
            res_alter["Dataset - target"] = res_alter["Dataset"] + " - " + res_alter["Target"]
            results.append(res_alter)

            res_def = pd.read_csv(Path(args.res_dir, f"mpp_acc_nfeature_{ds}.csv"), index_col=0)
            res_def = res_def.loc[res_def["Target var"].isin(valid_targets[ds])]
            res_def = res_def.assign(Dataset=ds).assign(Model="default")
            res_def["Dataset - target"] = res_def["Dataset"] + " - " + res_def["Target"]
            results.append(res_def)
        results = pd.concat(results, axis="index")

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
    nec_freq_file = Path(args.out_dir, f"alter_nec_freq_{dataset}.png")
    if (not nec_freq_file.exists()) or args.overwrite:
        nec_freq = pd.read_csv(Path(args.res_dir, f"mpp_alter_nec_freq_{dataset}.csv"), index_col=0)
        annot = nec_freq.map("{:.0%}".format).astype(str).replace("0%", "")
        nec_freq = nec_freq.mul(100)
        plt.figure(figsize=(nec_freq.shape[1]*1.2, nec_freq.shape[0]*0.5))
        ax = sns.heatmap(
            data=nec_freq, vmin=0, vmax=100, cbar=True, cmap=cmap_heat, annot=annot, fmt="",
            linewidth=.5, cbar_kws={"format": "%d%%", "label": "Probability"})
        ax.tick_params(axis="x", labelrotation=45, labeltop=True, labelbottom=False)
        xlabels = [target_names[label.get_text()] for label in ax.get_xticklabels()]
        ax.set_xticklabels(xlabels, ha="left")
        plt.savefig(nec_freq_file, bbox_inches="tight", dpi=500)
        plt.close()
