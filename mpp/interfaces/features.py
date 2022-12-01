from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import numpy as np
import nibabel as nib
import pandas as pd
import subprocess
from os import path, environ
import logging

from nipype.interfaces import fsl, freesurfer
import bct

from mpp import logger

base_dir = path.join(path.dirname(path.realpath(__file__)), '..')
logging.getLogger('datalad').setLevel(logging.WARNING)

### RSFC: compute Pearson's correlation between timeseries

class _RSFCInputSpec(BaseInterfaceInputSpec):
    tavg = traits.Dict(dtype=float, desc='parcellated timeseries')
    rs_skip = traits.Bool(desc='whether resting-state workflow should be skipped or not')

class _RSFCOutputSpec(TraitedSpec):
    rsfc = traits.Dict(dtype=float, desc='resting-state functional connectivity')

class RSFC(SimpleInterface):
    input_spec = _RSFCInputSpec
    output_spec = _RSFCOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.rs_skip:
            logger.warning('Resting-state workflow is skipped.')
        else:
            n_parcels = {1: 116, 2: 232, 3: 350, 4: 454}
            self._results['rsfc'] = {'level1': np.array([]), 'level2': np.array([]),
                                     'level3': np.array([]), 'level4': np.array([])}

            for level in range(4):
                rsfc_level = np.zeros((n_parcels[level+1], n_parcels[level+1], 4))
                runs = np.array([])

                for run in range(4):
                    key = 'run' + str(run+1) + '_level' + str(level+1)
                    if self.inputs.tavg[key].size:
                        rsfc = np.corrcoef(self.inputs.tavg[key])
                        rsfc = (0.5 * (np.log(1 + rsfc, where=~np.eye(rsfc.shape[0], dtype=bool)) 
                                - np.log(1 - rsfc, where=~np.eye(rsfc.shape[0], dtype=bool)))) # Fisher's z
                        for i in range(rsfc.shape[0]):
                            rsfc[i, i] = 0
                        rsfc_level[:, :, run] = rsfc
                        runs = np.concatenate([runs, [run]], axis=0)

                self._results['rsfc'][('level' + str(level+1))] = rsfc_level[:, :, runs.astype(int)].mean(axis=2)
                    
        return runtime

### DFC: compute autocorrelation-based dynamic connectivity

class _DFCInputSpec(BaseInterfaceInputSpec):
    tavg = traits.Dict(dtype=float, desc='parcellated timeseries')
    rs_skip = traits.Bool(desc='whether resting-state workflow should be skipped or not')

class _DFCOutputSpec(TraitedSpec):
    dfc = traits.Dict(dtype=float, desc='dynamic functional connectivity')

class DFC(SimpleInterface):
    input_spec = _DFCInputSpec
    output_spec = _DFCOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.rs_skip:
            logger.warning('Resting-state workflow is skipped.')
        else:
            n_parcels = {1: 116, 2: 232, 3: 350, 4: 454}
            self._results['dfc'] = {'level1': np.array([]), 'level2': np.array([]),
                                    'level3': np.array([]), 'level4': np.array([])}

            for level in range(4):
                dfc_level = np.zeros((n_parcels[level+1], n_parcels[level+1], 4))
                runs = np.array([])

                for run in range(4):
                    key = 'run' + str(run+1) + '_level' + str(level+1)
                    t_avg = self.inputs.tavg[key]
                    if t_avg.size:
                        y = t_avg[:, range(1, t_avg.shape[1])]
                        z = np.ones((t_avg.shape[0]+1, t_avg.shape[1]-1))
                        z[1:(t_avg.shape[0]+1), :] = t_avg[:, range(t_avg.shape[1]-1)]
                        b = np.linalg.lstsq((z @ z.T).T, (y @ z.T).T, rcond=None)[0].T
                        dfc_level[:, :, run] = b[:, range(1, b.shape[1])]
                        runs = np.concatenate([runs, [run]], axis=0)

                self._results['dfc'][('level' + str(level+1))] = dfc_level[:, :, runs.astype(int)].mean(axis=2)
                    
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
        if self.inputs.rs_skip:
            logger.warning('Resting-state workflow is skipped.')
        else:
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
        if self.inputs.myelin_skip:
            logger.warning('Myelin features are skipped.')
        else:
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
        if self.inputs.t1_skip:
            logger.warning('Morphometry featurs are skipped.')
        else:
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