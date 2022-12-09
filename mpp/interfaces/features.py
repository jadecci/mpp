from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import numpy as np
import nibabel as nib
import pandas as pd
import subprocess
from os import path, environ
import logging

from scipy.stats import zscore
from nipype.interfaces import fsl, freesurfer
import bct

from mpp.utilities.confounds import nuisance_conf_HCP

base_dir = path.join(path.dirname(path.realpath(__file__)), '..')
logging.getLogger('datalad').setLevel(logging.WARNING)

### RSFC: compute static and dynamic functional connectivity for resting-state

class _RSFCInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    rs_dir = traits.Str(desc='absolute path to installed subject MNINonLinear directory')
    rs_files = traits.Dict(desc='filenames of resting-state data')
    rs_skip = traits.Bool(desc='whether resting-state workflow should be skipped or not')

class _RSFCOutputSpec(TraitedSpec):
    rsfc = traits.Dict(dtype=float, desc='resting-state functional connectivity')
    dfc = traits.Dict(dtype=float, desc='dynamic functional connectivity')

class RSFC(SimpleInterface):
    input_spec = _RSFCInputSpec
    output_spec = _RSFCOutputSpec

    def _run_interface(self, runtime):
        if not self.inputs.rs_skip:
            eps = 7./3 - 4./3 - 1 # for checking non-brain voxels
            self._results['rsfc'] = {'level1': np.array([]), 'level2': np.array([]),
                                     'level3': np.array([]), 'level4': np.array([])}
            self._results['dfc'] = {'level1': np.array([]), 'level2': np.array([]),
                                    'level3': np.array([]), 'level4': np.array([])}
            runs = np.array([])

            for run in range(4):
                key_surf = 'run' + str(run+1) + '_surf'
                key_vol = 'run' + str(run+1) + '_vol'
                if self.inputs.rs_files[key_surf] and self.inputs.rs_files[key_vol]:
                    runs = np.concatenate([runs, [run]], axis=0)
                    t_surf = nib.load(self.inputs.rs_files[key_surf]).get_fdata()
                    t_vol = nib.load(self.inputs.rs_files[key_vol]).get_fdata()

                    if 'HCP' in self.inputs.dataset:
                        conf = nuisance_conf_HCP(self.inputs.dataset, self.inputs.rs_dir, self.inputs.rs_files[key_vol],
                                                 self.inputs.rs_files['wm_mask'], 
                                                 self.inputs.rs_files[('run' + str(run+1) + '_movement')])                  
                    regressors = np.concatenate([zscore(conf), np.ones((conf.shape[0], 1)), 
                                                np.linspace(-1, 1, num=conf.shape[0]).reshape((conf.shape[0], 1))],
                                                axis=1)
                    t_surf_resid = t_surf - np.dot(regressors, np.linalg.lstsq(regressors, t_surf, rcond=-1)[0])

                    for level in range(4):
                        parc_Sch_file = path.join(base_dir, 'data', 'atlas', ('Schaefer2018_' + str(level+1) + 
                                                                              '00Parcels_17Networks_order.dlabel.nii'))
                        parc_Sch = nib.load(parc_Sch_file).get_fdata()
                        parc_Mel_file = path.join(base_dir, 'data', 'atlas', 
                                                 ('Tian_Subcortex_S' + str(level+1) + '_3T.nii.gz'))
                        parc_Mel = nib.load(parc_Mel_file).get_fdata()
                        key = 'level' + str(level+1)
                        
                        mask = parc_Mel.nonzero()
                        t_vol_subcort = np.array([t_vol[mask[0][i], mask[1][i], mask[2][i], :] 
                                                  for i in range(mask[0].shape[0])])
                        t_vol_resid = t_vol_subcort.T - np.dot(regressors, 
                                                        np.linalg.lstsq(regressors, t_vol_subcort.T, rcond=-1)[0])

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
                            selected = selected[:, np.where(np.abs(selected.mean(axis=0))>=eps)[0]]
                            parc_vol[parcel-1, :] = selected.mean(axis=1)

                        tavg = np.concatenate([parc_surf, parc_vol], axis=0)
                        rsfc = np.corrcoef(tavg)
                        rsfc = (0.5 * (np.log(1 + rsfc, where=~np.eye(rsfc.shape[0], dtype=bool)) 
                                - np.log(1 - rsfc, where=~np.eye(rsfc.shape[0], dtype=bool)))) # Fisher's z
                        for i in range(rsfc.shape[0]):
                            rsfc[i, i] = 0
                        if self._results['rsfc'][key].size:
                            self._results['rsfc'][key] = self._results['rsfc'][key] + rsfc
                        else:
                            self._results['rsfc'][key] = rsfc

                        y = tavg[:, range(1, tavg.shape[1])]
                        z = np.ones((tavg.shape[0]+1, tavg.shape[1]-1))
                        z[1:(tavg.shape[0]+1), :] = tavg[:, range(tavg.shape[1]-1)]
                        b = np.linalg.lstsq((z @ z.T).T, (y @ z.T).T, rcond=None)[0].T
                        if self._results['dfc'][key].size:
                            self._results['dfc'][key] = self._results['dfc'][key] + b[:, range(1, b.shape[1])]
                        else:
                            self._results['dfc'][key] = b[:, range(1, b.shape[1])]

            for level in range(4):
                key = 'level' + str(level+1)
                self._results['rsfc'][key] = np.divide(self._results['rsfc'][key], len(runs))
                self._results['dfc'][key] = np.divide(self._results['dfc'][key], len(runs))
                    
        return runtime

### NetworkStats: compute network statistics based on RSFC

class _NetworkStatsInputSpec(BaseInterfaceInputSpec):
    rsfc = traits.Dict(dtype=float, desc='resting-state functional connectivity')
    rs_skip = traits.Bool(desc='whether resting-state workflow should be skipped or not')

class _NetworkStatsOutputSpec(TraitedSpec):
    rs_stats = traits.Dict(dtype=float, desc='dynamic functional connectivity')

class NetworkStats(SimpleInterface):
    input_spec = _NetworkStatsInputSpec
    output_spec = _NetworkStatsOutputSpec

    def _run_interface(self, runtime):
        if not self.inputs.rs_skip:
            self._results['rs_stats'] = {'level1_strength': np.array([]), 'level1_betweenness': np.array([]),
                                         'level1_participation': np.array([]), 'level1_efficiency': np.array([]),
                                         'level2_strength': np.array([]), 'level2_betweenness': np.array([]),
                                         'level2_participation': np.array([]), 'level2_efficiency': np.array([]),
                                         'level3_strength': np.array([]), 'level3_betweenness': np.array([]),
                                         'level3_participation': np.array([]), 'level3_efficiency': np.array([]),
                                         'level4_strength': np.array([]), 'level4_betweenness': np.array([]),
                                         'level4_participation': np.array([]), 'level4_efficiency': np.array([])}

            for level in range(4):
                rsfc = self.inputs.rsfc[('level' + str(level+1))]
                strength = bct.strengths_und(rsfc)
                betweenness = bct.betweenness_wei(rsfc)
                participation = bct.participation_coef(rsfc, bct.community_louvain(rsfc, B='negative_sym')[0])
                efficiency = bct.efficiency_wei(rsfc, local=True)

                self._results['rs_stats'][('level' + str(level+1) + '_strength')] = strength
                self._results['rs_stats'][('level' + str(level+1) + '_betweenness')] = betweenness
                self._results['rs_stats'][('level' + str(level+1) + '_participation')] = participation
                self._results['rs_stats'][('level' + str(level+1) + '_efficiency')] = efficiency

        return runtime

### MyelinEstimate: extract the myelin estimates from T1dividedbyT2 files

class _MyelineEstimateInputSpec(BaseInterfaceInputSpec):
    myelin_skip = traits.Bool(desc='whether myelin feature computation should be skipped or not')
    anat_files = traits.Dict(desc='filenames of anatomical data')

class _MyelineEstimateOutputSpec(TraitedSpec):
    myelin = traits.Dict(desc='myelin content estimates')

class MyelinEstimate(SimpleInterface):
    input_spec = _MyelineEstimateInputSpec
    output_spec = _MyelineEstimateOutputSpec

    def _run_interface(self, runtime):
        if not self.inputs.myelin_skip:
            self._results['myelin'] = {'level1': np.array([]), 'level2': np.array([]),
                                       'level3': np.array([]), 'level4': np.array([])}

            myelin_l = nib.load(self.inputs.anat_files['myelin_l']).agg_data()
            myelin_r = nib.load(self.inputs.anat_files['myelin_r']).agg_data()
            myelin_surf = np.hstack((myelin_l, myelin_r))
            myelin_vol = nib.load(self.inputs.anat_files['myelin_vol']).get_fdata()

            for level in range(4):
                parc_Sch_file = path.join(base_dir, 'data', 'atlas', 
                                          ('Schaefer2018_' + str(level+1) + '00Parcels_17Networks_order.dlabel.nii'))
                parc_Sch = nib.load(parc_Sch_file).get_fdata()
                parc_Mel_file = path.join(base_dir, 'data', 'atlas', ('Tian_Subcortex_S' + str(level+1) + '_3T.nii.gz'))
                parc_Mel = nib.load(parc_Mel_file).get_fdata()

                parc_surf = np.zeros(((level+1)*100))
                for parcel in range((level+1)*100):
                    selected = myelin_surf[np.where(parc_Sch==(parcel+1))[1]]
                    selected = selected[~np.isnan(selected)]
                    parc_surf[parcel] = selected.mean()

                parc_Mel_mask = parc_Mel.nonzero()
                parc_Mel = parc_Mel[parc_Mel.nonzero()]
                myelin_vol_masked = np.array([myelin_vol[parc_Mel_mask[0][i], parc_Mel_mask[1][i], parc_Mel_mask[2][i]]
                                             for i in range(parc_Mel_mask[0].shape[0])])
                parcels = np.unique(parc_Mel).astype(int)
                parc_vol = np.zeros((parcels.shape[0]))
                for parcel in parcels:
                    selected = myelin_vol_masked[np.where(parc_Mel==(parcel))[0]]
                    selected = selected[~np.isnan(selected)]
                    parc_vol[parcel-1] = selected.mean()
                
                key = 'level' + str(level+1)
                self._results['myelin'][key] = np.hstack([parc_surf, parc_vol])

        return runtime

### Morphometry: extract morphometry features

class _MorphometryInputSpec(BaseInterfaceInputSpec):
    t1_skip = traits.Bool(desc='whether morphometry feature computation should be skipped or not')
    anat_dir = traits.Str(desc='absolute path to installed subject T1w directory')
    anat_files = traits.Dict(desc='filenames of anatomical data')
    subject = traits.Str(desc='subject ID')

class _MorphometryOutputSpec(TraitedSpec):
    morph = traits.Dict(desc='morphometry features')

class Morphometry(SimpleInterface):
    input_spec = _MorphometryInputSpec
    output_spec = _MorphometryOutputSpec

    def _run_interface(self, runtime):
        if not self.inputs.t1_skip:
            self._results['morph'] = {'level1_GMV': np.array([]), 'level1_CS': np.array([]), 'level1_CT': np.array([]),
                                      'level2_GMV': np.array([]), 'level2_CS': np.array([]), 'level2_CT': np.array([]),
                                      'level3_GMV': np.array([]), 'level3_CS': np.array([]), 'level3_CT': np.array([]),
                                      'level4_GMV': np.array([]), 'level4_CS': np.array([]), 'level4_CT': np.array([])}

            for level in range(4): 
                stats_surf = pd.DataFrame()
                for hemi in ['lh', 'rh']:
                    annot_file = path.join(base_dir, 'data', 'label', 
                                           (hemi + '.Schaefer2018_' + str(level+1) + '00Parcels_17Networks_order.annot'))
                    hemi_table = hemi + '.fs_stats'
                    subprocess.run(['mris_anatomical_stats', '-a', annot_file, '-noglobal', '-f', hemi_table, 
                           self.inputs.subject, hemi], env=dict(environ, **{'SUBJECTS_DIR': self.inputs.anat_dir}),
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                    hemi_stats = pd.read_table(hemi_table, header=0, skiprows=np.arange(51), delim_whitespace=True)
                    hemi_stats.drop([0], inplace=True) # exclude medial wall
                    stats_surf = stats_surf.append(hemi_stats)
                self._results['morph'][('level' + str(level+1) + '_CS')] = stats_surf['SurfArea'].values
                self._results['morph'][('level' + str(level+1) + '_CT')] = stats_surf['ThickAvg'].values

                seg_file = path.join(base_dir, 'data', 'atlas', ('Tian_Subcortex_S' + str(level+1) + '_3T.nii.gz'))
                seg_up_file = 'S' + str(level) + '_upsampled.nii.gz'
                flt = fsl.FLIRT()
                flt.inputs.in_file = seg_file
                flt.inputs.reference = self.inputs.anat_files['t1_vol']
                flt.inputs.out_file = seg_up_file
                flt.inputs.apply_isoxfm = 0.8
                flt.inputs.interp = 'nearestneighbour'
                flt.terminal_output = 'file_split'
                flt.run()

                sub_table = 'subcortex.stats'
                ss = freesurfer.SegStats()
                ss.inputs.segmentation_file = seg_up_file
                ss.inputs.in_file = self.inputs.anat_files['t1_vol']
                ss.inputs.summary_file = sub_table
                ss.inputs.subjects_dir = self.inputs.anat_dir
                ss.terminal_output = 'file_split'
                ss.run()

                stats_vol = pd.read_table(sub_table, header=0, skiprows=np.arange(50), delim_whitespace=True)
                stats_vol.drop([0], inplace=True)

                key = 'level' + str(level+1) + '_GMV'
                self._results['morph'][key] = np.concatenate((stats_surf['GrayVol'].values, 
                                                              stats_vol['Volume_mm3'].values))
                                    
        return runtime