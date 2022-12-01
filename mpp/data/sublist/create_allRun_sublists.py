import pandas as pd
from os import path

data_dir = path.join(path.dirname(path.realpath(__file__)), '..')


### HCP-YA
hcpya_pheno_file = path.join(data_dir, 'phenotype', 'unrestricted_jadecci_11_13_2018_6_44_10.csv')
hcpya_scores = ['Subject', 'T1_Count', '3T_RS-fMRI_Count', '3T_tMRI_PctCompl', '3T_dMRI_Compl']
hcpya_data = pd.read_csv(hcpya_pheno_file, usecols=hcpya_scores)[hcpya_scores]
hcpya_drop = hcpya_data.loc[(hcpya_data['T1_Count']==0) & 
                            (hcpya_data['3T_RS-fMRI_Count']==0) & 
                            (hcpya_data['3T_tMRI_PctCompl']==0) & 
                            (hcpya_data['3T_dMRI_Compl']==False)].index
hcpya_data.drop(hcpya_drop, inplace=True)
hcpya_out_file = path.join(data_dir, 'sublist', 'HCP-YA_allRun.csv')
hcpya_data['Subject'].to_csv(hcpya_out_file, header=False, index=False)

### HCP-A
hcpa_pheno_file = path.join(data_dir, 'phenotype', 'HCP-A', 'ndar_subject01.txt')
hcpa_data = pd.read_table(hcpa_pheno_file, sep='\t', header=0, skiprows=[1], usecols=['src_subject_id'])
hcpa_out_file = path.join(data_dir, 'sublist', 'HCP-A_allRun.csv')
hcpa_data.to_csv(hcpa_out_file, header=False, index=False)

### HCP-D
hcpd_pheno_file = path.join(data_dir, 'phenotype', 'HCP-D', 'ndar_subject01.txt')
hcpd_data = pd.read_table(hcpd_pheno_file, sep='\t', header=0, skiprows=[1], usecols=['src_subject_id'])
hcpd_out_file = path.join(data_dir, 'sublist', 'HCP-D_allRun.csv')
hcpd_data.to_csv(hcpd_out_file, header=False, index=False)