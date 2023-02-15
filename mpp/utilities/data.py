import h5py
import pandas as pd
from os import path
import pathlib
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

def read_h5(h5_file, dataset):
    with h5py.File(h5_file, 'r') as f:
        ds = f[dataset]
        data = ds[()]

    return data

def pheno_HCP(dataset, pheno_dir, pheno_name, sublist):
    if dataset == 'HCP-YA':
        col_names = {'totalcogcomp': 'CogTotalComp_AgeAdj'}
        unres_file = sorted(pathlib.Path(pheno_dir).glob('unrestricted_*.csv'))[0]
        pheno_data = pd.read_csv(unres_file, usecols=['Subject', col_names[pheno_name]], dtype={'Subject': str, 
                                 col_names[pheno_name]: float})['Subject', col_names[pheno_name]]

    elif dataset == 'HCP-A' or dataset == 'HCP-D':
        pheno_file = {'totalcogcomp': 'cogcomp01.txt'}
        pheno_cols = {'totalcogcomp': 30}
        col_names = {'totalcogcomp': 'nih_totalcogcomp_ageadjusted'}

        pheno_data = pd.read_table(path.join(pheno_dir, pheno_file[pheno_name]), sep='\t', header=0, skiprows=[1],
                                   usecols=[4, pheno_cols[pheno_name]], dtype={'src_subject_id': str, 
                                   col_names[pheno_name]: float})['Subject', col_names[pheno_name]]

    pheno_data.columns = ['subject', pheno_name]
    pheno_data = pheno_data.dropna().drop_duplicates(subset='subject')
    pheno_data = pheno_data[pheno_data['subject'].isin(sublist)]

    sublist = pheno_data['subject'].to_list()
    pheno_dict = pheno_data.set_index('subject').squeeze().to_dict()

    pheno_data[pheno_name] = pheno_data[pheno_name].sample(frac=1, ignore_index=True)
    pheno_dict_perm = pheno_data.set_index('subject').squeeze().to_dict()

    return sublist, pheno_dict, pheno_dict_perm

        