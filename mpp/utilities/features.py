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
