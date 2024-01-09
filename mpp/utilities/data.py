from pathlib import Path
from typing import Union

import h5py
import numpy as np

from mpp.exceptions import DatasetError
from mpp.utilities.features import diffusion_mapping_sub, score_sub


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
    sc_count = read_h5(feature_file, f'/sc/count/level{level}')
    sc_length = read_h5(feature_file, f'/sc/length/level{level}')
    sc_count_strength = read_h5(feature_file, f'/sc_count_stats/strength/level{level}')
    sc_count_betweenness = read_h5(feature_file, f'/sc_count_stats/betweenness/level{level}')
    sc_count_participation = read_h5(feature_file, f'/sc_count_stats/participation/level{level}')
    sc_count_efficiency = read_h5(feature_file, f'/sc_count_stats/efficiency/level{level}')
    sc_length_strength = read_h5(feature_file, f'/sc_length_stats/strength/level{level}')
    sc_length_betweenness = read_h5(feature_file, f'/sc_length_stats/betweenness/level{level}')
    sc_length_participation = read_h5(feature_file, f'/sc_length_stats/participation/level{level}')
    sc_length_efficiency = read_h5(feature_file, f'/sc_length_stats/efficiency/level{level}')

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
        rsfc, dfc, efc, gradients, tfc, strength, betweenness, participation, efficiency, myelin,
        gmv, cs, ct, ac_gmv, ac_cs, ac_ct, sc_count, sc_length, sc_count_strength,
        sc_count_betweenness, sc_count_participation, sc_count_efficiency, sc_length_strength,
        sc_length_betweenness, sc_length_participation, sc_length_efficiency)


def cv_extract_all_features(
    subjects: list, sublists: dict, features_dir: dict, level: str, embeddings: dict,
    params: dict, repeat: int, phenotypes: dict,
    permutation: bool = False) -> tuple[np.ndarray, np.ndarray,]:
    y = np.zeros(len(subjects))
    x_all = np.array([])

    for i, subject in enumerate(subjects):
        x = cv_extract_subject_data(
            sublists, subject, features_dir, level, permutation, embeddings, params, repeat)
        x_task = np.concatenate(([
            x[4][:, :, i][np.triu_indices_from(x[4][:, :, i], k=1)]
            for i in range(x[4].shape[2])]))
        x_curr = np.concatenate((
            x[0][np.triu_indices_from(x[0], k=1)], x[1][np.triu_indices_from(x[1], k=1)],
            x[2][np.triu_indices_from(x[2], k=1)], x[3].flatten(), x_task, x[5], x[6], x[7],
            x[8], x[9], x[10], x[11], x[12], x[13][np.triu_indices_from(x[13], k=1)],
            x[14][np.triu_indices_from(x[14], k=1)], x[15][np.triu_indices_from(x[15], k=1)],
            x[16][np.triu_indices_from(x[16], k=1)], x[17][np.triu_indices_from(x[17], k=1)],
            x[19], x[20], x[21], x[22], x[23], x[24], x[25], x[26]))
        if i == 0:
            x_all = x_curr[np.newaxis, ...]
        else:
            x_all = np.vstack((x_all, x_curr[np.newaxis, ...]))
        y[i] = phenotypes[subjects[i]]

    return x_all, y
