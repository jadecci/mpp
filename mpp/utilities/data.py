from typing import Union
import logging
from pathlib import Path

import numpy as np
import h5py
import pandas as pd

from mpp.exceptions import DatasetError
from mpp.utilities.features import diffusion_mapping_sub, score_sub

logging.getLogger('datalad').setLevel(logging.WARNING)


def write_h5(h5_file: Union[Path, str], dataset: str, data: np.ndarray, overwrite: bool) -> None:
    with h5py.File(h5_file, 'a') as f:
        try:
            if data.shape:
                ds = f.require_dataset(
                    dataset, shape=data.shape, dtype=data.dtype, data=data, chunks=True,
                    maxshape=(None,)*data.ndim)
            else:
                ds = f.require_dataset(dataset, shape=data.shape, dtype=data.dtype, data=data)
        except TypeError:
            if overwrite:
                ds = f[dataset]
                if data.shape:
                    ds.resize(data.shape)
                ds.write_direct(data)
            else:
                raise TypeError(
                    'Existing dataset with different data shape found. '
                    'Use overwrite=True to overwrite existing data.')


def read_h5(h5_file: Union[Path, str], dataset: str) -> np.ndarray:
    with h5py.File(h5_file, 'r') as f:
        ds = f[dataset]
        data = ds[()]

    if not isinstance(data, (dict, np.ndarray)):
        raise TypeError("'read_h5' expects dict or np.ndarray data")

    return data


def pheno_hcp(
        dataset: str, pheno_dir: Union[Path, str], pheno_name: str,
        sublist: list) -> tuple[list, dict, dict]:
    if dataset == 'HCP-YA':
        col_names = {'totalcogcomp': 'CogTotalComp_AgeAdj'}
        unres_file = sorted(Path(pheno_dir).glob('unrestricted_*.csv'))[0]
        pheno_data = pd.read_csv(
            unres_file, usecols=['Subject', col_names[pheno_name]],
            dtype={'Subject': str, col_names[pheno_name]: float})[[
                'Subject', col_names[pheno_name]]]

    elif dataset == 'HCP-A' or dataset == 'HCP-D':
        pheno_file = {'totalcogcomp': 'cogcomp01.txt'}
        pheno_cols = {'totalcogcomp': 30}
        col_names = {'totalcogcomp': 'nih_totalcogcomp_ageadjusted'}

        pheno_data = pd.read_table(
            Path(pheno_dir, pheno_file[pheno_name]), sep='\t', header=0, skiprows=[1],
            usecols=[4, pheno_cols[pheno_name]],
            dtype={'src_subject_id': str, col_names[pheno_name]: float})[[
                'src_subject_id', col_names[pheno_name]]]

    else:
        raise DatasetError()

    pheno_data.columns = ['subject', pheno_name]
    pheno_data = pheno_data.dropna().drop_duplicates(subset='subject').reset_index(drop=True)
    pheno_data = pheno_data[pheno_data['subject'].isin(sublist)]

    sublist_out = pheno_data['subject'].to_list()
    pheno_dict = pheno_data.set_index('subject').squeeze().to_dict()

    pheno_data[pheno_name] = pheno_data[pheno_name].sample(frac=1, ignore_index=True)
    pheno_dict_perm = pheno_data.set_index('subject').squeeze().to_dict()

    return sublist_out, pheno_dict, pheno_dict_perm


def cv_extract_data(
        sublists: dict, features_dir: dict, subjects: list, repeat: int, level: str,
        embeddings: dict, params: dict, phenotypes: dict, permutation: bool = False,
        selected_features: Union[np.ndarray, None] = None) -> tuple[np.ndarray, ...]:
    y = np.zeros(len(subjects))
    x_all = np.zeros(len(subjects))

    for i, subject in enumerate(subjects):
        dataset = [key for key in sublists if subject in sublists[key]][0]
        if dataset == 'HCP-A' or 'HCP-D':
            feature_file = Path(
                features_dir[dataset], f'{dataset}_{subject}_V1_MR.h5')
        else:
            feature_file = Path(
                features_dir[dataset], f'{dataset}_{subject}.h5')

        rsfc = read_h5(feature_file, f'/rsfc/level{level}')
        dfc = read_h5(feature_file, f'/dfc/level{level}')
        strength = read_h5(feature_file, f'/network_stats/strength/level{level}')
        betweenness = read_h5(feature_file, f'/network_stats/betweenness/level{level}')
        participation = read_h5(feature_file, f'/network_stats/participation/level{level}')
        efficiency = read_h5(feature_file, f'/network_stats/efficiency/level{level}')
        # tfc = read_h5(feature_file, f'/tfc/level{self.inputs.level}')
        myelin = read_h5(feature_file, f'/myelin/level{level}')
        gmv = read_h5(feature_file, f'/morphometry/GMV/level{level}')
        cs = read_h5(feature_file, f'/morphometry/CS/level{level}')
        ct = read_h5(feature_file, f'/morphometry/CT/level{level}')

        if permutation:
            gradients = diffusion_mapping_sub(embeddings[f'repeat{repeat}'], rsfc)
            ac_gmv = score_sub(params[f'repeat{repeat}'], gmv)
            ac_cs = score_sub(params[f'repeat{repeat}'], cs)
            ac_ct = score_sub(params[f'repeat{repeat}'], ct)
        else:
            gradients = diffusion_mapping_sub(embeddings['embedding'], rsfc)
            ac_gmv = score_sub(params['params'], gmv)
            ac_cs = score_sub(params['params'], cs)
            ac_ct = score_sub(params['params'], ct)

        x = np.vstack((
            rsfc.mean(axis=2), dfc.mean(axis=2), strength, betweenness, participation, efficiency,
            # tfc.reshape(tfc.shape[0], tfc.shape[1]*tfc.shape[2]).T,
            myelin, gmv, np.pad(cs, (0, len(gmv) - len(cs))), np.pad(ct, (0, len(gmv) - len(ct))),
            gradients, ac_gmv, np.hstack((ac_cs, np.zeros((ac_cs.shape[0], len(gmv) - len(cs))))),
            np.hstack((ac_ct, np.zeros((ac_cs.shape[0], len(gmv) - len(ct)))))))
        # TODO: diffusion features
        x_all = x if i == 0 else np.dstack((x_all.T, x.T)).T  # N x F x R
        y[i] = phenotypes[subjects[i]]

    if selected_features is not None:
        x_old = x_all
        x_all = None
        for region in range(selected_features.shape[0]):
            if selected_features[region, :].sum() != 0:
                x_region = x_old[:, selected_features[region, :], region]
                if x_all is None:
                    x_all = x_region
                else:
                    x_all = np.hstack((x_all, x_region))

    return x_all, y
