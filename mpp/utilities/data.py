from typing import Union
import logging
from pathlib import Path

import numpy as np
import h5py
import pandas as pd

from mpp.exceptions import DatasetError
from mpp.utilities.features import diffusion_mapping_sub, score_sub

logging.getLogger('datalad').setLevel(logging.WARNING)

task_runs = {
    'HCP-YA': ['tfMRI_EMOTION_LR', 'tfMRI_EMOTION_RL', 'tfMRI_GAMBLING_LR', 'tfMRI_GAMBLING_RL',
               'tfMRI_LANGUAGE_LR', 'tfMRI_LANGUAGE_RL', 'tfMRI_MOTOR_LR', 'tfMRI_MOTOR_RL',
               'tfMRI_RELATIONAL_LR', 'tfMRI_RELATIONAL_RL', 'tfMRI_SOCIAL_LR', 'tfMRI_SOCIAL_RL',
               'tfMRI_WM_LR', 'tfMRI_WM_RL'],
    'HCP-A': ['tfMRI_CARIT_PA', 'tfMRI_FACENAME_PA', 'tfMRI_VISMOTOR_PA'],
    'HCP-D': ['tfMRI_CARIT_AP', 'tfMRI_CARIT_PA', 'tfMRI_EMOTION_PA', 'tfMRI_GUESSING_AP',
              'tfMRI_GUESSING_PA']}


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


def read_h5(
        h5_file: Union[Path, str], dataset: str, impute: bool = False,
        impute_shape: Union[tuple, None] = None, impute_val: float = 0) -> np.ndarray:
    with h5py.File(h5_file, 'r') as f:
        try:
            ds = f[dataset]
            data = ds[()]
        except KeyError:
            if impute and impute_shape is not None:
                data = np.ones(impute_shape) * impute_val
            elif impute_shape is None:
                raise TypeError("'impute_shape' must be a tuple (not None) if impute is True")
            else:
                raise KeyError

    if not isinstance(data, (dict, np.ndarray)):
        raise TypeError("'read_h5' expects dict or np.ndarray data")

    return data


def pheno_hcp(
        dataset: str, pheno_dir: Union[Path, str], pheno_name: str,
        sublist: list) -> tuple[list, dict, dict]:
    if dataset == 'HCP-YA':
        col_names = {
            'totalcogcomp': 'CogTotalComp_AgeAdj', 'fluidcogcomp': 'CogFluidComp_AgeAdj',
            'crycogcomp': 'CogCrystalComp_AgeAdj'}
        unres_file = sorted(Path(pheno_dir).glob('unrestricted_*.csv'))[0]
        pheno_data = pd.read_csv(
            unres_file, usecols=['Subject', col_names[pheno_name]],
            dtype={'Subject': str, col_names[pheno_name]: float})[[
                'Subject', col_names[pheno_name]]]

    elif dataset == 'HCP-A' or dataset == 'HCP-D':
        if dataset == 'HCP-A':
            pheno_cols = {'totalcogcomp': 30, 'fluidcogcomp': 14, 'crycogcomp': 18}
        else:
            pheno_cols = {'totalcogcomp': 18, 'fluidcogcomp': 9, 'crycogcomp': 12}
        pheno_file = {
            'totalcogcomp': 'cogcomp01.txt', 'fluidcogcomp': 'cogcomp01.txt',
            'crycogcomp': 'cogcomp01.txt'}
        col_names = {
            'totalcogcomp': 'nih_totalcogcomp_ageadjusted',
            'fluidcogcomp': 'nih_fluidcogcomp_ageadjusted',
            'crycogcomp': 'nih_crycogcomp_ageadjusted'}

        pheno_data = pd.read_table(
            Path(pheno_dir, pheno_file[pheno_name]), sep='\t', header=0, skiprows=[1],
            usecols=[4, pheno_cols[pheno_name]],
            dtype={'src_subject_id': str, col_names[pheno_name]: float})[[
                'src_subject_id', col_names[pheno_name]]]

    else:
        raise DatasetError()

    pheno_data.columns = ['subject', pheno_name]
    pheno_data = pheno_data.dropna().drop_duplicates(subset='subject')
    pheno_data = pheno_data[pheno_data['subject'].isin(sublist)].reset_index(drop=True)

    sublist_out = pheno_data['subject'].to_list()
    pheno_dict = pheno_data.set_index('subject').squeeze().to_dict()

    pheno_data[pheno_name] = pheno_data[pheno_name].sample(frac=1, ignore_index=True)
    pheno_dict_perm = pheno_data.set_index('subject').squeeze().to_dict()

    return sublist_out, pheno_dict, pheno_dict_perm


def cv_extract_subject_data(
        sublists: dict, subject: str, features_dir: dict, level: str, permutation: bool,
        embeddings: dict, params: dict, repeat: int) -> tuple[np.ndarray, ...]:
    dataset = [key for key in sublists if subject in sublists[key]][0]
    if dataset in ['HCP-A', 'HCP-D']:
        feature_file = Path(
            features_dir[dataset], f'{dataset}_{subject}_V1_MR.h5')
    else:
        feature_file = Path(
            features_dir[dataset], f'{dataset}_{subject}.h5')

    rsfc = read_h5(feature_file, f'/rsfc/level{level}')
    dfc = read_h5(feature_file, f'/dfc/level{level}')
    efc = read_h5(feature_file, f'/efc/level{level}')
    strength = read_h5(feature_file, f'/network_stats/strength/level{level}')
    betweenness = read_h5(feature_file, f'/network_stats/betweenness/level{level}')
    participation = read_h5(feature_file, f'/network_stats/participation/level{level}')
    efficiency = read_h5(feature_file, f'/network_stats/efficiency/level{level}')
    myelin = read_h5(feature_file, f'/myelin/level{level}')
    gmv = read_h5(feature_file, f'/morphometry/GMV/level{level}')
    cs = read_h5(feature_file, f'/morphometry/CS/level{level}')
    ct = read_h5(feature_file, f'/morphometry/CT/level{level}')

    tfc = np.zeros((rsfc.shape[0], rsfc.shape[1], len(task_runs[dataset])))
    for run, task in enumerate(task_runs[dataset]):
        tfc[:, :, run] = read_h5(feature_file, f'/tfc/{task}/level{level}')
        #    feature_file, f'/tfc/{task}/level{level}', impute=True, impute_shape=rsfc.shape)

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

    return (
        rsfc, dfc, efc, tfc, strength, betweenness, participation, efficiency, myelin, gmv, cs, ct,
        gradients, ac_gmv, ac_cs, ac_ct)


def cv_extract_data(
        sublists: dict, features_dir: dict, subjects: list, repeat: int, level: str,
        embeddings: dict, params: dict, phenotypes: dict,
        permutation: bool = False) -> tuple[np.ndarray, ...]:
    y = np.zeros(len(subjects))
    x_all = np.array([])

    for i, subject in enumerate(subjects):
        rsfc, dfc, efc, tfc, strength, betweenness, participation, efficiency, myelin, gmv, cs, \
            ct, gradients, ac_gmv, ac_cs, ac_ct = cv_extract_subject_data(
                sublists, subject, features_dir, level, permutation, embeddings, params, repeat)
        x = np.vstack((
            rsfc, dfc, efc, tfc.reshape(tfc.shape[0], tfc.shape[1]*tfc.shape[2]).T,
            strength, betweenness, participation, efficiency,
            myelin, gmv, np.pad(cs, (0, len(gmv) - len(cs))), np.pad(ct, (0, len(gmv) - len(ct))),
            gradients, ac_gmv, np.hstack((ac_cs, np.zeros((ac_cs.shape[0], len(gmv) - len(cs))))),
            np.hstack((ac_ct, np.zeros((ac_cs.shape[0], len(gmv) - len(ct)))))))
        # TODO: diffusion features
        x_all = x if i == 0 else np.dstack((x_all.T, x.T)).T  # N x F x R
        y[i] = phenotypes[subjects[i]]

    return x_all, y
