from pathlib import Path
import argparse

import datalad.api as dl
import pandas as pd

phenos = [
    "totalcogcomp", "crycogcomp", "fluidcogcomp", "cardsort", "flanker", "reading", "picvocab",
    "procspeed", "listsort", "anger", "fear", "sadness", "posaffect", "emotsupp", "friendship",
    "loneliness", "neoffi_n", "neoffi_e", "neoffi_o", "neoffi_a", "neoffi_c"]

parser = argparse.ArgumentParser(
    description="Collect and results for fusion CCA analysis",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument(
    "--out_dir", type=Path, dest="out_dir", required=True, help="Absolute path to output directory")
parser.add_argument(
    "--work_dir", type=Path, dest="work_dir", required=True, help="Absolute path to work directory")
args = parser.parse_args()

args.out_dir.mkdir(parents=True, exist_ok=True)

# Install dataset with fusion CCA results
res_url = "git@gin.g-node.org:/jadecci/fusion_cca_data.git"
root_res_dir = Path(args.work_dir, "fusion_cca_data")
dl.install(root_res_dir, source=res_url)
res_dir = Path(root_res_dir, "fusion_cca_results")

for dataset in ["HCP-D", "HCP-YA", "HCP-A"]:
    for pheno in phenos:
        for ftype in ["conn", "region", "dti"]:
            out_file = Path(args.out_dir, f"fusion_cca_sig_{dataset}_{pheno}_{ftype}.csv")
            res_sig = []
            for fold in range(100):
                res_file = Path(res_dir, f"fusion_cca_{dataset}_{pheno}_fold{fold}_{ftype}.csv")
                if res_file.is_symlink():
                    dl.get(res_file, dataset=root_res_dir)
                    res_curr = pd.read_csv(res_file, header=0, index_col=[0])
                    res_sig.append(res_curr.loc[res_curr["P-value"] < 0.05])
                    dl.drop(res_file, dataset=root_res_dir)
            pd.concat(res_sig, axis="index").reset_index(drop=True).to_csv(out_file)
            print(f"Collected results for {dataset} {pheno} {ftype}-features")

dl.remove(dataset=root_res_dir, reckless="kill")
