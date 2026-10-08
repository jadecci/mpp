from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser(
    description="Plot collected confound model results",
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

# Confound model accuracy
conf_file = Path(args.out_dir, "conf_acc.png")
if (not conf_file.exists()) or args.overwrite:
    results = []
    for ds in dataset_order:
        res_curr = pd.read_csv(Path(args.res_dir, f"mpp_acc_{ds}.csv"), index_col=0)
        res_curr = res_curr.loc[res_curr["Prediction type"] == "confounds"]
        results.append(res_curr.assign(Dataset=ds))
    results = pd.concat(results, axis="index")

    g = sns.catplot(
        data=results, kind="box", y="Target", x="Accuracy", hue="Target", col="Dataset",
        row="Accuracy type", col_order=dataset_order, orient="h", palette=cmap,
        height=10, aspect=0.7, showfliers=False, showcaps=False, boxprops={"linewidth": 0},
        whiskerprops={"color": "lightgray", "linewidth": 1.5}, medianprops={"linewidth": 2.5},
        legend=False, sharey=True, sharex=False)
    for ax in g.axes.flat:
        ax.axvline(color="lightgray", linestyle="--")
    plt.savefig(conf_file, bbox_inches="tight", dpi=500)
    plt.close()
