import sys
from typing import Union
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore
from scipy.ndimage import binary_erosion
from sklearn.metrics import pairwise_distances
from mapalign.embed import compute_diffusion_map
from statsmodels.formula.api import ols
from rdcmpy import RegressionDCM
from sklearn.linear_model import LinearRegression

from mpp.exceptions import DatasetError

base_dir = Path(__file__).resolve().parent.parent
logging.getLogger('datalad').setLevel(logging.WARNING)


def pheno_conf_hcp(
        dataset: str, pheno_dir: Union[Path, str], features_dir: Union[Path, str],
        sublist: list) -> tuple[list, dict]:
    # primary vairables
    if dataset == 'HCP-YA':
        unres_file = sorted(Path(pheno_dir).glob('unrestricted_*.csv'))[0]
        res_file = sorted(Path(pheno_dir).glob('RESTRICTED_*.csv'))[0]
        unres_conf = pd.read_csv(
            unres_file, usecols=['Subject', 'Gender', 'FS_BrainSeg_Vol', 'FS_IntraCranial_Vol'],
            dtype={
                'Subject': str, 'Gender': str, 'FS_BrainSeg_Vol': float,
                'FS_IntraCranial_Vol': float})
        res_conf = pd.read_csv(
            res_file, usecols=['Subject', 'Age_in_Yrs', 'Handedness'],
            dtype={'Subject': str, 'Age_in_Yrs': int, 'Handedness': int})
        conf = unres_conf.merge(res_conf, on='Subject', how='inner').dropna()
        conf = conf[[
            'Subject', 'Age_in_Yrs', 'Gender', 'Handedness', 'FS_BrainSeg_Vol',
            'FS_IntraCranial_Vol']]

    elif dataset == 'HCP-A' or dataset == 'HCP-D':
        conf = pd.read_table(
            Path(pheno_dir, 'ssaga_cover_demo01.txt'), sep='\t', header=0, skiprows=[1],
            usecols=[4, 5, 7], dtype={'src_subject_id': str, 'interview_age': int, 'sex': str})
        conf = conf.merge(pd.read_table(
            Path(pheno_dir, 'edinburgh_hand01.txt'), sep='\t', header=0, skiprows=[1],
            usecols=[5, 70], dtype={'src_subject_id': str, 'hcp_handedness_score': int}),
            on='src_subject_id', how='inner')

        brainseg_vols = []
        icv_vols = []
        for subject in conf['src_subject_id']:
            astats_file = Path(features_dir, f'{dataset}_astats', f'{subject}_V1_MR.txt')
            aseg_stats = pd.read_csv(str(astats_file), sep='\t', index_col=0)
            brainseg_vols.append(aseg_stats['BrainSegVol'][0])
            icv_vols.append(aseg_stats['EstimatedTotalIntraCranialVol'][0])
        conf['brainseg_vol'] = brainseg_vols
        conf['icv_vol'] = icv_vols

        conf = conf[[
            'src_subject_id', 'interview_age', 'sex', 'hcp_handedness_score', 'brainseg_vol',
            'icv_vol']]

    else:
        raise DatasetError()

    conf.columns = ['subject', 'age', 'gender', 'handedness', 'brainseg_vol', 'icv_vol']
    conf = conf.dropna().drop_duplicates(subset='subject')
    conf = conf[conf['subject'].isin(sublist)]

    # gender coding: 1 for Female, 2 for Male
    conf['gender'] = [1 if item == 'F' else 2 for item in conf['gender']]
    # secondary variables
    conf['age2'] = np.power(conf['age'], 2)
    conf['ageGender'] = conf['age'] * conf['gender']
    conf['age2Gender'] = conf['age2'] * conf['gender']

    sublist_out = conf['subject'].to_list()
    conf_dict = conf.set_index('subject').to_dict()

    return sublist_out, conf_dict


def diffusion_mapping(image_features: dict, sublist: list, input_key: str) -> np.ndarray:
    n_parcels = image_features[sublist[0]][input_key].shape[0]
    rsfc = np.zeros((n_parcels, n_parcels, len(sublist)))
    for i in range(len(sublist)):
        rsfc[:, :, i] = image_features[sublist[i]][input_key]

    # transform by tanh and threshold RSFC at 90th percentile
    rsfc_thresh = np.tanh(rsfc.mean(axis=2))
    for i in range(rsfc_thresh.shape[0]):
        rsfc_thresh[i, rsfc_thresh[i, :] < np.percentile(rsfc_thresh[i, :], 90)] = 0
    rsfc_thresh[rsfc_thresh < 0] = 0  # there should be very few negatives after thresholding

    affinity = 1 - pairwise_distances(rsfc_thresh, metric='cosine')
    embed = compute_diffusion_map(affinity, alpha=0.5)

    return embed


def diffusion_mapping_sub(embed: np.ndarray, sub_rsfc: np.ndarray) -> np.ndarray:
    return embed.T @ sub_rsfc


def score(
        image_features: dict, sublist: list, input_key: str) -> pd.DataFrame:
    # see https://github.com/katielavigne/score/blob/main/score.py
    n_parcels = len(image_features[sublist[0]][input_key])
    features = pd.DataFrame(columns=range(n_parcels), index=sublist)
    for i in range(len(sublist)):
        features.loc[sublist[i]] = image_features[sublist[i]][input_key]
    features = features.join(pd.DataFrame({'mean': features.mean(axis=1)}))
    features[features.columns] = features[features.columns].apply(pd.to_numeric)

    ac = np.zeros((n_parcels, n_parcels, len(sublist)))
    params = pd.DataFrame()
    for i in range(n_parcels):
        for j in range(n_parcels):
            results = ols(f'features[{i}] ~ features[{j}] + mean', data=features).fit()
            ac[i, j, :] = results.resid
            params[f'{i}_{j}'] = [
                results.params['Intercept'], results.params[f'features[{j}]'],
                results.params['mean']]

    return params


def score_sub(params: pd.DataFrame, sub_features: np.ndarray) -> np.ndarray:
    # see https://github.com/katielavigne/score/blob/main/score.py
    mean_features = sub_features.mean()
    n_parcels = sub_features.shape[0]
    ac = np.zeros((n_parcels, n_parcels))

    for i in range(n_parcels):
        for j in range(n_parcels):
            params_curr = params[f'{i}_{j}']
            ac[i, j] = (
                    params_curr[0] + params_curr[1] * sub_features[j] +
                    params_curr[2] * mean_features)
    return ac


def pheno_reg_conf(
        train_y: np.ndarray, train_conf: np.ndarray, test_y: np.ndarray,
        test_conf: np.ndarray) -> tuple[np.ndarray, ...]:
    conf_reg = LinearRegression()
    conf_reg.fit(train_conf, train_y)
    train_y_resid = train_y - conf_reg.predict(train_conf)
    test_y_resid = test_y - conf_reg.predict(test_conf)

    return train_y_resid, test_y_resid
