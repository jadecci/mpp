import numpy as np
import pandas as pd
import nibabel as nib
import datalad.api as dl
from os import path
import logging

from scipy.ndimage import binary_erosion

logging.getLogger('datalad').setLevel(logging.WARNING)

def nuisance_conf_HCP(dataset, rs_dir, rs_file, atlas_file, motion_file):
    csf_code = np.array([4, 5, 14, 15, 24, 31, 43, 44, 63, 250, 251, 252, 253, 254, 255]) - 1

    data = nib.load(rs_file).get_fdata()
    data = data.reshape((data.shape[0]*data.shape[1]*data.shape[2], data.shape[3]))
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