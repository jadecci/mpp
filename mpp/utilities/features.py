import numpy as np
import pandas as pd
import nibabel as nib
from os import path
import sys
import logging

from scipy.stats import zscore
from scipy.ndimage import binary_erosion

base_dir = path.join(path.dirname(path.realpath(__file__)), '..')
logging.getLogger('datalad').setLevel(logging.WARNING)

def fc(t_surf, t_vol, dataset, run, func_files, sfc_dict, dfc_dict=None):
    if 'HCP' in dataset:
        conf = nuisance_conf_HCP(t_vol, func_files['wm_mask'], func_files[f'{run}_movement'])
    regressors = np.concatenate([zscore(conf), np.ones((conf.shape[0], 1)), 
                                 np.linspace(-1, 1, num=conf.shape[0]).reshape((conf.shape[0], 1))], axis=1)
    t_surf_resid = t_surf - np.dot(regressors, np.linalg.lstsq(regressors, t_surf, rcond=-1)[0])

    for level in range(4):
        key = f'level{level+1}'

        parc_Sch_file = path.join(base_dir, 'data', 'atlas', 
                                  f'Schaefer2018_{level+1}00Parcels_17Networks_order.dlabel.nii')
        parc_Sch = nib.load(parc_Sch_file).get_fdata()
        parc_Mel_file = path.join(base_dir, 'data', 'atlas', f'Tian_Subcortex_S{level+1}_3T.nii.gz')
        parc_Mel = nib.load(parc_Mel_file).get_fdata()
        
        mask = parc_Mel.nonzero()
        t_vol_subcort = np.array([t_vol[mask[0][i], mask[1][i], mask[2][i], :] for i in range(mask[0].shape[0])])
        t_vol_resid = t_vol_subcort.T - np.dot(regressors, np.linalg.lstsq(regressors, t_vol_subcort.T, rcond=-1)[0])

        t_surf = t_surf_resid[:, range(parc_Sch.shape[1])]
        parc_surf = np.zeros(((level+1)*100, t_surf.shape[0]))
        for parcel in range((level+1)*100):
            selected = t_surf[:, np.where(parc_Sch==(parcel+1))[1]]
            selected = selected[:, ~np.isnan(selected[0, :])]
            parc_surf[parcel, :] = selected.mean(axis=1)

        parcels = np.unique(parc_Mel[mask]).astype(int)
        parc_vol = np.zeros((parcels.shape[0], t_vol_resid.shape[0]))
        for parcel in parcels:
            selected = t_vol_resid[:, np.where(parc_Mel[mask]==(parcel))[0]]
            selected = selected[:, ~np.isnan(selected[0, :])]
            selected = selected[:, np.where(np.abs(selected.mean(axis=0))>=sys.float_info.epsilon)[0]]
            parc_vol[parcel-1, :] = selected.mean(axis=1)

        tavg = np.concatenate([parc_surf, parc_vol], axis=0)

        # static FC 
        sfc = np.corrcoef(tavg)
        sfc = (0.5 * (np.log(1 + sfc, where=~np.eye(sfc.shape[0], dtype=bool)) 
               - np.log(1 - sfc, where=~np.eye(sfc.shape[0], dtype=bool)))) # Fisher's z
        sfc[np.diag_indices_from(sfc)] = 0
        if sfc_dict[key].size:
            sfc_dict[key] = sfc_dict[key] + sfc
        else:
            sfc_dict[key] = sfc

        # dynamic FC (optional)
        if not dfc_dict == None:
            y = tavg[:, range(1, tavg.shape[1])]
            z = np.ones((tavg.shape[0]+1, tavg.shape[1]-1))
            z[1:(tavg.shape[0]+1), :] = tavg[:, range(tavg.shape[1]-1)]
            b = np.linalg.lstsq((z @ z.T).T, (y @ z.T).T, rcond=None)[0].T
            if dfc_dict[key].size:
                dfc_dict[key] = dfc_dict[key] + b[:, range(1, b.shape[1])]
            else:
                dfc_dict[key] = b[:, range(1, b.shape[1])]

    return sfc_dict, dfc_dict


def nuisance_conf_HCP(t_vol, atlas_file, motion_file):
    csf_code = np.array([4, 5, 14, 15, 24, 31, 43, 44, 63, 250, 251, 252, 253, 254, 255]) - 1
    data = t_vol.reshape((t_vol.shape[0]*t_vol.shape[1]*t_vol.shape[2], t_vol.shape[3]))
    atlas = nib.load(atlas_file).get_fdata()
    atlas = atlas.reshape((data.shape[0]))

    # WM & CSF confounds
    wm_ind = np.where(atlas >= 3000)[0]
    wm_mask = np.zeros(atlas.shape)
    wm_mask[wm_ind] = 1
    wm_mask = binary_erosion(wm_mask).reshape((atlas.shape))
    wm_signal = data[np.where(wm_mask == 1)[0], :].mean(axis=0)
    wm_diff = np.diff(wm_signal, prepend=wm_signal[0])
    csf_signal = data[[i for i in range(len(atlas)) if atlas[i] in csf_code]].mean(axis=0)
    csf_diff = np.diff(csf_signal, prepend=csf_signal[0])

    # motion parameters
    motion = pd.read_table(motion_file, sep='  ', header=None, engine='python')
    motion = motion.join(np.power(motion, 2), lsuffix='motion', rsuffix='motion2')

    conf = motion.assign(wm=wm_signal).assign(wmdiff=wm_diff).assign(csf=csf_signal).assign(csfdiff=csf_diff)

    return conf