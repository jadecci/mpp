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

from mpp.exceptions import DatasetError

base_dir = Path(__file__).resolve().parent.parent
logging.getLogger('datalad').setLevel(logging.WARNING)


def parcellate(
        t_surf: np.ndarray, t_vol: np.ndarray, dataset: str, func_files: dict) -> dict:
    if dataset in ['HCP-YA', 'HCP-A', 'HCP-D']:
        conf = nuisance_conf_hcp(t_vol, func_files['atlas_mask'])
    else:
        raise DatasetError()

    regressors = np.concatenate((
        zscore(conf), np.ones((conf.shape[0], 1)),
        np.linspace(-1, 1, num=conf.shape[0]).reshape((conf.shape[0], 1))), axis=1)
    t_surf_resid = t_surf - np.dot(regressors, np.linalg.lstsq(regressors, t_surf, rcond=-1)[0])

    tavg_dict = {}
    for level in range(4):
        key = f'level{level+1}'
        parc_sch_file, _, _, parc_mel_file, _ = atlas_files(Path(base_dir, 'data'), level)
        parc_sch = nib.load(parc_sch_file).get_fdata()
        parc_mel = nib.load(parc_mel_file).get_fdata()

        mask = parc_mel.nonzero()
        t_vol_subcort = np.array(
            [t_vol[mask[0][i], mask[1][i], mask[2][i], :] for i in range(mask[0].shape[0])])
        t_vol_resid = (
                t_vol_subcort.T -
                np.dot(regressors, np.linalg.lstsq(regressors, t_vol_subcort.T, rcond=-1)[0]))

        t_surf = t_surf_resid[:, range(parc_sch.shape[1])]
        parc_surf = np.zeros(((level+1)*100, t_surf.shape[0]))
        for parcel in range((level+1)*100):
            selected = t_surf[:, np.where(parc_sch == (parcel + 1))[1]]
            selected = selected[:, ~np.isnan(selected[0, :])]
            parc_surf[parcel, :] = selected.mean(axis=1)

        parcels = np.unique(parc_mel[mask]).astype(int)
        parc_vol = np.zeros((parcels.shape[0], t_vol_resid.shape[0]))
        for parcel in parcels:
            selected = t_vol_resid[:, np.where(parc_mel[mask] == (parcel))[0]]
            selected = selected[:, ~np.isnan(selected[0, :])]
            selected = selected[
                :, np.where(np.abs(selected.mean(axis=0)) >= sys.float_info.epsilon)[0]]
            parc_vol[parcel-1, :] = selected.mean(axis=1)

        tavg_dict[key] = np.vstack((parc_surf, parc_vol))

    return tavg_dict


def fc(tavg_dict: dict, t_rep: Union[float, None] = None) -> tuple[dict, dict, dict]:
    sfc_dict = {}
    dfc_dict = {}
    efc_dict = {}
    for level in range(4):
        key = f'level{level+1}'
        tavg = tavg_dict[key]

        # static FC (Fisher's z excluding diagonals)
        sfc = np.corrcoef(tavg_dict[key])
        sfc = (
                0.5 * (np.log(1 + sfc, where=~np.eye(sfc.shape[0], dtype=bool)) -
                       np.log(1 - sfc, where=~np.eye(sfc.shape[0], dtype=bool))))
        sfc[np.diag_indices_from(sfc)] = 0
        sfc_dict[key] = sfc

        # dynamic FC: 1st order ARR model
        # see https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/fMRI_dynamics/Liegeois2017_Surrogates/CBIG_RL2017_ar_mls.m # noqa: E501
        y = tavg[:, range(1, tavg.shape[1])]
        z = np.ones((tavg.shape[0] + 1, tavg.shape[1] - 1))
        z[1:(tavg.shape[0] + 1), :] = tavg[:, range(tavg.shape[1] - 1)]
        b = np.linalg.lstsq((z @ z.T).T, (y @ z.T).T, rcond=None)[0].T
        dfc_dict[key] = b[:, range(1, b.shape[1])]

        # effective FC: regression DCM
        if t_rep is not None:
            rdcm = RegressionDCM(data=tavg.T, t_rep=t_rep)
            rdcm.estimate()
            params = rdcm.get_params()
            efc_dict[key] = params['mu_connectivity']

    return sfc_dict, dfc_dict, efc_dict


def nuisance_conf_hcp(t_vol: np.ndarray, atlas_file: Union[Path, str]) -> np.ndarray:
    # Atlas labels follow FreeSurferColorLUT
    # see https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    csf_code = np.array([4, 5, 14, 15, 24, 31, 43, 44, 63, 250, 251, 252, 253, 254, 255]) - 1
    data = t_vol.reshape((t_vol.shape[0]*t_vol.shape[1]*t_vol.shape[2], t_vol.shape[3]))
    atlas = nib.load(atlas_file).get_fdata()
    atlas = atlas.reshape((data.shape[0]))

    # gloabl signals
    global_signal = data[np.where(atlas != 0)[0], :].mean(axis=0)
    global_diff = np.diff(global_signal, prepend=global_signal[0])

    # WM signals
    wm_ind = np.where(atlas >= 3000)[0]
    wm_mask = np.zeros(atlas.shape)
    wm_mask[wm_ind] = 1
    wm_mask = binary_erosion(wm_mask).reshape((atlas.shape))
    wm_signal = data[np.where(wm_mask == 1)[0], :].mean(axis=0)
    wm_diff = np.diff(wm_signal, prepend=wm_signal[0])

    # CSF signals
    csf_signal = data[[i for i in range(len(atlas)) if atlas[i] in csf_code]].mean(axis=0)
    csf_diff = np.diff(csf_signal, prepend=csf_signal[0])

    # We will not regress out motion parameters for FIX denoised data
    # see https://www.mail-archive.com/hcp-users@humanconnectome.org/msg02957.html
    # motion = pd.read_table(motion_file, sep='  ', header=None, engine='python')
    # motion = motion.join(np.power(motion, 2), lsuffix='motion', rsuffix='motion2')

    conf = np.vstack((global_signal, global_diff, wm_signal, wm_diff, csf_signal, csf_diff)).T

    return conf


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


def add_subdir(sub_dir: str, subject: str, fs_dir: str) -> str:
    from pathlib import Path
    from os import getenv, symlink
    from shutil import copytree

    Path(sub_dir).mkdir(parents=True, exist_ok=True)
    if not Path(sub_dir, subject).is_symlink():
        symlink(fs_dir, Path(sub_dir, subject))
    if not Path(sub_dir, 'fsaverage').is_dir():
        copytree(
            Path(getenv('FREESURFER_HOME'), 'subjects', 'fsaverage'),
            Path(sub_dir, 'fsaverage'),
            dirs_exist_ok=True)

    return sub_dir


def atlas_files(data_dir: str, level: int) -> tuple[str, str, str, str, int]:
    from pathlib import Path

    parc_sch = Path(
        data_dir, 'atlas', f'Schaefer2018_{level+1}00Parcels_17Networks_order.dlabel.nii')
    lh_annot_sch = Path(
        data_dir, 'label', f'lh.Schaefer2018_{level+1}00Parcels_17Networks_order.annot')
    rh_annot_sch = Path(
        data_dir, 'label', f'rh.Schaefer2018_{level+1}00Parcels_17Networks_order.annot')
    parc_mel = Path(data_dir, 'atlas', f'Tian_Subcortex_S{level+1}_3T.nii.gz')

    return str(parc_sch), str(lh_annot_sch), str(rh_annot_sch), str(parc_mel), level


def add_annot(sub_dir: str, subject: str, lh_annot: str, rh_annot: str) -> str:
    import shutil
    from pathlib import Path

    if Path(lh_annot).is_file():
        shutil.copyfile(lh_annot, Path(sub_dir, subject, 'label', Path(lh_annot).name))
    if Path(rh_annot).is_file():
        shutil.copyfile(rh_annot, Path(sub_dir, subject, 'label', Path(rh_annot).name))

    annot_name = str(Path(lh_annot).name).split('.')[1:-1]
    annot_name = '.'.join(annot_name)
    annot_args = f'--annot {annot_name}'

    return annot_args

