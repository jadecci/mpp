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
    'HCP-YA': [
        'tfMRI_EMOTION', 'tfMRI_GAMBLING', 'tfMRI_LANGUAGE', 'tfMRI_MOTOR', 'tfMRI_RELATIONA',
        'tfMRI_SOCIAL', 'tfMRI_WM'],
    'HCP-A': ['tfMRI_CARIT_PA', 'tfMRI_FACENAME_PA', 'tfMRI_VISMOTOR_PA'],
    'HCP-D': ['tfMRI_CARIT', 'tfMRI_EMOTION_PA', 'tfMRI_GUESSING']}


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
            'crycogcomp': 'CogCrystalComp_AgeAdj', 'cardsort': 'CardSort_AgeAdj',
            'flanker': 'Flanker_AgeAdj', 'reading': 'ReadEng_AgeAdj', 'picvocab': 'PicVocab_AgeAdj',
            'procspeed': 'ProcSpeed_AgeAdj', 'ddisc': 'DDisc_AUC_40K',
            'listsort': 'ListSort_AgeAdj', 'emotrecog': 'ER40_CR', 'anger': 'AngAffect_Unadj',
            'fear': 'FearAffect_Unadj', 'sadness': 'Sadness_Unadj', 'posaffect': 'PosAffect_Unadj',
            'emotsupp': 'EmotSupp_Unadj', 'friendship': 'Friendship_Unadj',
            'loneliness': 'Loneliness_Unadj', 'endurance': 'Endurance_AgeAdj',
            'gaitspeed': 'GaitSpeed_Comp', 'strength': 'Strength_AgeAdj', 'neoffi_n': 'NEOFAC_N',
            'neoffi_e': 'NEOFAC_E', 'neoffi_o': 'NEOFAC_O', 'neoffi_a': 'NEOFAC_A',
            'neoffi_c': 'NEOFAC_C'}
        unres_file = sorted(Path(pheno_dir).glob('unrestricted_*.csv'))[0]
        pheno_data = pd.read_csv(
            unres_file, usecols=['Subject', col_names[pheno_name]],
            dtype={'Subject': str, col_names[pheno_name]: float})[[
                'Subject', col_names[pheno_name]]]

    elif dataset == 'HCP-A' or dataset == 'HCP-D':
        if dataset == 'HCP-A':
            pheno_cols = {
                'totalcogcomp': 30, 'fluidcogcomp': 14, 'crycogcomp': 18, 'cardsort': 76,
                'flanker': 55, 'reading': 10, 'picvocab': 10, 'procspeed': 145, 'ddisc': 133,
                'listsort': 136, 'emotrecog': 8, 'anger': 31, 'fear': 38, 'sadness': 37,
                'posaffect': 157, 'emotsupp': 12, 'friendship': 12, 'loneliness': 23,
                'endurance': 16, 'gaitspeed': 30, 'strength': 22, 'neoffi_n': 77, 'neoffi_e': 76,
                'neoffi_o': 78, 'neoffi_a': 74, 'neoffi_c': 75}
        else:
            pheno_cols = {
                'totalcogcomp': 18, 'fluidcogcomp': 9, 'crycogcomp': 12, 'cardsort': 41,
                'flanker': 11, 'reading': 10, 'picvocab': 10, 'procspeed': 10, 'ddisc': 22,
                'listsort': 36, 'emotrecog': 8, 'anger': 29, 'fear': 28, 'sadness': 26,
                'posaffect': 66, 'emotsupp': 11, 'friendship': 10, 'loneliness': 11,
                'endurance': 13, 'gaitspeed': 18, 'strength': 14, 'neoffi_n': 71, 'neoffi_e': 70,
                'neoffi_o': 72, 'neoffi_a': 68, 'neoffi_c': 69}
        pheno_file = {
            'totalcogcomp': 'cogcomp01.txt', 'fluidcogcomp': 'cogcomp01.txt',
            'crycogcomp': 'cogcomp01.txt', 'cardsort': 'dccs01.txt', 'flanker': 'flanker01.txt',
            'reading': 'orrt01.txt', 'picvocab': 'tpvt01.txt', 'procspeed': 'pcps01.txt',
            'ddisc': 'deldisk01.txt', 'listsort': 'lswmt01.txt', 'emotrecog': 'er4001.txt',
            'anger': 'prang01.txt', 'fear': 'preda01.txt', 'sadness': 'predd01.txt',
            'posaffect': 'tlbx_wellbeing01.txt', 'emotsupp': 'tlbx_emsup01.txt',
            'friendship': 'tlbx_friend01.txt', 'loneliness': 'prsi01.txt',
            'endurance': 'tlbx_motor01.txt', 'gaitspeed': 'tlbx_motor01.txt',
            'strength': 'tlbx_motor01.txt', 'neoffi_n': 'nffi01.txt', 'neoffi_e': 'nffi01.txt',
            'neoffi_o': 'nffi01.txt', 'neoffi_a': 'nffi01.txt', 'neoffi_c': 'nffi01.txt'}
        col_names = {
            'totalcogcomp': 'nih_totalcogcomp_ageadjusted',
            'fluidcogcomp': 'nih_fluidcogcomp_ageadjusted',
            'crycogcomp': 'nih_crycogcomp_ageadjusted', 'cardsort': 'nih_dccs_ageadjusted',
            'flanker': 'nih_flanker_ageadjusted', 'reading': 'read_acss', 'picvocab': 'tpvt_acss',
            'procspeed': 'nih_patterncomp_ageadjusted', 'ddisc': 'auc_40000',
            'listsort': 'age_corrected_standard_score', 'emotrecog': 'er40_c_cr',
            'anger': 'anger_ts', 'fear': 'anx_ts', 'sadness': 'add_ts','posaffect': 'tlbxpa_ts',
            'emotsupp': 'nih_tlbx_tscore', 'friendship': 'nih_tlbx_tscore', 'loneliness': 'soil_ts',
            'endurance': 'end_2m_standardsc', 'gaitspeed': 'loco_comscore',
            'strength': 'grip_standardsc_dom', 'neoffi_n': 'neo2_score_ne',
            'neoffi_e': 'neo2_score_ex', 'neoffi_o': 'neo2_score_op', 'neoffi_a': 'neo2_score_ag',
            'neoffi_c': 'neo2_score_co'}

        pheno_data = pd.read_table(
            Path(pheno_dir, pheno_file[pheno_name]), sep='\t', header=0, skiprows=[1],
            usecols=[4, pheno_cols[pheno_name]],
            dtype={'src_subject_id': str, col_names[pheno_name]: float})[[
                'src_subject_id', col_names[pheno_name]]]

        if dataset == 'HCP-A' and pheno_name == 'fluidcogcomp':
            pheno_data.drop(
                index=pheno_data.loc[pheno_data[col_names[pheno_name]] == 999].index, inplace=True)

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
