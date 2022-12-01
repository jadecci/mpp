from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import nibabel as nib
import numpy as np
from os import path
import logging

from scipy.stats import zscore

from mpp import logger
from mpp.utilities.confounds import nuisance_conf_HCP

base_dir = path.join(path.dirname(path.realpath(__file__)), '..')
logging.getLogger('datalad').setLevel(logging.WARNING)

### NuisanceReg: regress out nuisance regressors from resting-state timeseries

class _NuisanceRegInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    rs_dir = traits.Str(desc='absolute path to installed subject MNINonLinear directory')
    rs_files = traits.Dict(desc='filenames of resting-state data')
    rs_skip = traits.Bool(desc='whether resting-state workflow should be skipped or not')

class _NuisanceRegOutputSpec(TraitedSpec):
    t_surf_resid = traits.Dict(desc='residual timeseries on surface after nuisance regression')
    t_vol_resid = traits.Dict(desc='residual timeseries in volume after nuisance regression')

class NuisanceReg(SimpleInterface):
    input_spec = _NuisanceRegInputSpec
    output_spec = _NuisanceRegOutputSpec

    def _run_interface(self, runtime):
        if not self.inputs.rs_skip:
            self._results['t_surf_resid'] = {'run1': np.array([]), 'run2': np.array([]), 'run3': np.array([]), 
                                             'run4': np.array([])}
            self._results['t_vol_resid'] = {'run1_level1': np.array([]), 'run1_level2': np.array([]),
                                            'run1_level3': np.array([]), 'run1_level4': np.array([]),
                                            'run2_level1': np.array([]), 'run2_level2': np.array([]),
                                            'run2_level3': np.array([]), 'run2_level4': np.array([]),
                                            'run3_level1': np.array([]), 'run3_level2': np.array([]),
                                            'run3_level3': np.array([]), 'run3_level4': np.array([]),
                                            'run4_level1': np.array([]), 'run4_level2': np.array([]),
                                            'run4_level3': np.array([]), 'run4_level4': np.array([])}

            for run in range(4):
                logger.info(f'Extracing imaging confounds for run {run+1}')
                key_surf = 'run' + str(run+1) + '_surf'
                key_vol = 'run' + str(run+1) + '_vol'
                key_move = 'run' + str(run+1) + '_movement'
                if self.inputs.rs_files[key_surf] and self.inputs.rs_files[key_vol]:
                    t_surf = nib.load(self.inputs.rs_files[key_surf]).get_fdata()
                    t_vol = nib.load(self.inputs.rs_files[key_vol]).get_fdata()
                    if 'HCP' in self.inputs.dataset:
                        conf = nuisance_conf_HCP(self.inputs.dataset, self.inputs.rs_dir, self.inputs.rs_files[key_vol],
                                                 self.inputs.rs_files['wm_mask'], self.inputs.rs_files[key_move])
                    
                    regressors = np.concatenate([zscore(conf), np.ones((conf.shape[0], 1)), 
                                                np.linspace(-1, 1, num=conf.shape[0]).reshape((conf.shape[0], 1))],
                                                axis=1)
                    t_surf_resid = t_surf - np.dot(regressors, np.linalg.lstsq(regressors, t_surf, rcond=-1)[0])
                    self._results['t_surf_resid'][('run' + str(run))] = t_surf_resid

                    # for volumetric data, only take the corresponding subcortical voxels
                    for level in range(4):
                        logger.info(f'Regressing out nuisance confounds at level {level+1}')
                        mask_file = path.join(base_dir, 'data', 'atlas', 
                                              ('Tian_Subcortex_S' + str(level+1) + '_3T.nii.gz'))
                        mask = nib.load(mask_file).get_fdata().nonzero()
                        t_vol_subcort = np.array([t_vol[mask[0][i], mask[1][i], mask[2][i], :] 
                                                  for i in range(mask[0].shape[0])])
                        t_vol_resid = t_vol_subcort.T - np.dot(regressors, np.linalg.lstsq(regressors, t_vol_subcort.T,
                                                                                           rcond=-1)[0])
                        self._results['t_vol_resid'][('run' + str(run+1) + '_level' + str(level+1))] = t_vol_resid

        return runtime

### ParcellateTimeseries: compute average timeseries in each parcel for resting-state data

class _ParcellateTimeseriesInputSpec(BaseInterfaceInputSpec):
    t_surf_resid = traits.Dict(desc='residual timeseries on surface after nuisance regression')
    t_vol_resid = traits.Dict(desc='residual timeseries in volume after nuisance regression')
    rs_skip = traits.Bool(desc='whether resting-state workflow should be skipped or not')

class _ParcellateTimeseriesOutputSpec(TraitedSpec):
    tavg = traits.Dict(dtype=float, desc='parcellated timeseries')

class ParcellateTimeseries(SimpleInterface):
    input_spec = _ParcellateTimeseriesInputSpec
    output_spec = _ParcellateTimeseriesOutputSpec

    def _run_interface(self, runtime):
        if not self.inputs.rs_skip:
            eps = 7./3 - 4./3 - 1 # for checking non-brain voxels
            self._results['tavg'] = {'run1_level1': np.array([]), 'run1_level2': np.array([]),
                                     'run1_level3': np.array([]), 'run1_level4': np.array([]),
                                     'run2_level1': np.array([]), 'run2_level2': np.array([]),
                                     'run2_level3': np.array([]), 'run2_level4': np.array([]),
                                     'run3_level1': np.array([]), 'run3_level2': np.array([]),
                                     'run3_level3': np.array([]), 'run3_level4': np.array([]),
                                     'run4_level1': np.array([]), 'run4_level2': np.array([]),
                                     'run4_level3': np.array([]), 'run4_level4': np.array([])}

            for run in range(4):
                for level in range(4):
                    logger.info(f'Parcellating timeseries for run {run+1} at level {level+1}')
                    key_surf = 'run' + str(run+1)
                    key_vol = 'run' + str(run+1) + '_level' + str(level+1)
                    if self.inputs.t_surf_resid[key_surf].size and self.inputs.t_vol_resid[key_vol].size:
                        parc_Sch_file = path.join(base_dir, 'data', 'atlas', 
                                                ('Schaefer2018_' + str(level+1) + 
                                                 '00Parcels_17Networks_order.dlabel.nii'))
                        parc_Sch = nib.load(parc_Sch_file).get_fdata()
                        parc_Mel_file = path.join(base_dir, 'data', 'atlas', 
                                                 ('Tian_Subcortex_S' + str(level+1) + '_3T.nii.gz'))
                        parc_Mel = nib.load(parc_Mel_file).get_fdata()

                        t_surf = self.inputs.t_surf_resid[key_surf][:, range(parc_Sch.shape[1])]
                        parc_surf = np.zeros(((level+1)*100, t_surf.shape[0]))
                        for parcel in range((level+1)*100):
                            selected = t_surf[:, np.where(parc_Sch==(parcel+1))[1]]
                            selected = selected[:, ~np.isnan(selected[0, :])]
                            parc_surf[parcel, :] = selected.mean(axis=1)

                        t_vol = self.inputs.t_vol_resid[key_vol]
                        parc_Mel = parc_Mel[parc_Mel.nonzero()]
                        parcels = np.unique(parc_Mel).astype(int)
                        parc_vol = np.zeros((parcels.shape[0], t_vol.shape[0]))
                        for parcel in parcels:
                            selected = t_vol[:, np.where(parc_Mel==(parcel))[0]]
                            selected = selected[:, ~np.isnan(selected[0, :])]
                            selected = selected[:, np.where(np.abs(selected.mean(axis=0))>=eps)[0]]
                            parc_vol[parcel-1, :] = selected.mean(axis=1)
                        
                        key = 'run' + str(run+1) + '_level' + str(level+1)
                        self._results['tavg'][key] = np.concatenate([parc_surf, parc_vol], axis=0)
            
        return runtime
