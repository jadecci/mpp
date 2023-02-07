import h5py
import pandas as pd
from os import path
import logging

logging.getLogger('datalad').setLevel(logging.WARNING)

def write_h5(h5_file, dataset, data, overwrite):
    with h5py.File(h5_file, 'a') as f:
                    try:
                        ds = f.require_dataset(dataset, shape=data.shape, dtype=data.dtype, data=data, chunks=True,
                                               maxshape=(None,)*data.ndim)
                    except TypeError:
                        if overwrite:
                            ds = f[dataset]
                            ds.resize(data.shape)
                            ds.write_direct(data)

def pheno_HCP(dataset, pheno_dir, pheno_name, sublist, pheno_dict):
    if dataset == 'HCP-YA':
        col_names = {'totalcogcomp': 'CogTotalComp_AgeAdj'}
    elif dataset == 'HCP-A' or dataset == 'HCP-D':
        pheno_file = {'totalcogcomp': 'cogcomp01.txt'}
        pheno_cols = {'totalcogcomp': 30}
        col_names = {'totalcogcomp': 'nih_totalcogcomp_ageadjusted'}

        pheno_data = pd.read_table(path.join(pheno_dir, pheno_file[pheno_name]), sep='\t', header=0, skiprows=[1],
                                   usecols=[4, pheno_cols[pheno_name]], 
                                   dtype={'src_subject_id': str, col_names[pheno_name]: float})
        pheno_data = pheno_data.dropna().drop_duplicates(subset='src_subject_id')
        pheno_data = pheno_data[pheno_data['src_subject_id'].isin(sublist)]

        sublist = pheno_data['src_subject_id'].to_list()
        pheno_dict = pheno_data.set_index('src_subject_id').squeeze().to_dict()

        return sublist, pheno_dict

        