from pathlib import Path
import argparse
import pandas as pd

parser = argparse.ArgumentParser(
    description="generate subject lists for HCP datasets",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("pheno_dir", type=Path, help="absolute path to phenotype directory")
parser.add_argument("out_dir", type=Path, help="absolute path to output directory")
parser.add_argument("hcpa_exclude", type=Path, help="absolute path to HCP-A exclusion list")
parser.add_argument("hcpd_exclude", type=Path, help="absolute path to HCP-D exclusion list")
parser.add_argument("hcpya_exclude", type=Path, help="absolute path to HCP-YA exclusion list")
args = parser.parse_args()

# HCP-YA
hcpya_pheno_file = Path(args.pheno_dir, "unrestricted_hcpya.csv")
hcpya_scores = ["Subject", "T1_Count", "3T_RS-fMRI_Count", "3T_tMRI_PctCompl", "3T_dMRI_PctCompl"]
hcpya_data = pd.read_csv(hcpya_pheno_file, usecols=hcpya_scores)[hcpya_scores]
hcpya_exclude = pd.read_csv(args.hcpya_excude, header=None).squeeze().to_list()
hcpya_data.drop(hcpya_data.loc[hcpya_data["Subject"].isin(hcpya_exclude)].index, inplace=True)
hcpya_data.drop(hcpya_data.loc[hcpya_data["T1_Count"] == 0].index, inplace=True)
hcpya_data.drop(hcpya_data.loc[hcpya_data["3T_RS-fMRI_Count"] == 0].index, inplace=True)
hcpya_data.drop(hcpya_data.loc[hcpya_data["3T_tMRI_PctCompl"] != 100].index, inplace=True)
hcpya_data.drop(hcpya_data.loc[hcpya_data["3T_dMRI_PctCompl"] != 100].index, inplace=True)
hcpya_out_file = Path(args.out_dir, "HCP-YA_allRun.csv")
hcpya_data["Subject"].to_csv(hcpya_out_file, header=False, index=False)

# HCP-A
hcpa_pheno_file = Path(args.pheno_dir, "HCP-A", "ndar_subject01.txt")
hcpa_data = pd.read_table(
    hcpa_pheno_file, sep="\t", header=0, skiprows=[1], usecols=["src_subject_id"])
hcpa_data["src_subject_id"] = hcpa_data["src_subject_id"].str.cat(["_V1_MR"]*len(hcpa_data))
hcpa_exclude = pd.read_csv(args.hcpa_exclude, header=None).squeeze().to_list()
hcpa_data.drop(hcpa_data.loc[hcpa_data["src_subject_id"].isin(hcpa_exclude)].index, inplace=True)
hcpa_out_file = Path(args.out_dir, "HCP-A_allRun.csv")
hcpa_data.to_csv(hcpa_out_file, header=False, index=False)

# HCP-D
hcpd_pheno_file = Path(args.pheno_dir, "HCP-D", "ndar_subject01.txt")
hcpd_data = pd.read_table(
    hcpd_pheno_file, sep="\t", header=0, skiprows=[1], usecols=["src_subject_id"])
hcpd_data["src_subject_id"] = hcpd_data["src_subject_id"].str.cat(["_V1_MR"]*len(hcpd_data))
hcpd_exclude = pd.read_csv(args.hcpd_exclude, header=None).squeeze().to_list()
hcpd_data.drop(hcpd_data.loc[hcpd_data["src_subject_id"].isin(hcpd_exclude)].index, inplace=True)
hcpd_out_file = Path(args.out_dir, "HCP-D_allRun.csv")
hcpd_data.to_csv(hcpd_out_file, header=False, index=False)
