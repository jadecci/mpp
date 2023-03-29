from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import numpy as np
import nibabel as nib
import pandas as pd
import subprocess
from os import environ
from pathlib import Path

from nipype.interfaces import fsl, freesurfer
import bct

from mpp.utilities.features import fc, diffusion_mapping, score
from mpp.utilities.data import read_h5

base_dir = Path(__file__).resolve().parent.parent

### RSFC: compute static and dynamic functional connectivity for resting-state

class _RSFCInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    rs_runs = traits.List(mandatory=True, desc='resting-state run names')
    rs_files = traits.Dict(mandatory=True, dtype=Path, desc='filenames of resting-state data')
    hcpd_b_runs = traits.Int(0, desc='number of b runs added for HCP-D subject')

class _RSFCOutputSpec(TraitedSpec):
    rsfc = traits.Dict(dtype=float, desc='resting-state functional connectivity')
    dfc = traits.Dict(dtype=float, desc='dynamic functional connectivity')

class RSFC(SimpleInterface):
    input_spec = _RSFCInputSpec
    output_spec = _RSFCOutputSpec

    def _run_interface(self, runtime):
        self._results['rsfc'] = {'level1': np.array([]), 'level2': np.array([]),
                                 'level3': np.array([]), 'level4': np.array([])}
        self._results['dfc'] = {'level1': np.array([]), 'level2': np.array([]),
                                'level3': np.array([]), 'level4': np.array([])}

        n_runs = len(self.inputs.rs_runs) + self.inputs.hcpd_b_runs
        for i in range(n_runs):
            if self.inputs.dataset == 'HCP-D' and i >= 4:
                run = self.inputs.rs_runs[i-3]
                key_surf = f'{run}_surfb'
                key_vol = f'{run}_volb'
            else:
                run = self.inputs.rs_runs[i]
                key_surf = f'{run}_surf'
                key_vol = f'{run}_vol'

            if self.inputs.rs_files[key_surf] and self.inputs.rs_files[key_vol]:
                t_surf = nib.load(self.inputs.rs_files[key_surf]).get_fdata()
                t_vol = nib.load(self.inputs.rs_files[key_vol]).get_fdata()          
                self._results['rsfc'], self._results['dfc'] = fc(t_surf, t_vol, self.inputs.dataset,
                                                                 self.inputs.rs_files, self._results['rsfc'], 
                                                                 self._results['dfc'])
                    
        return runtime

### NetworkStats: compute network statistics based on RSFC

class _NetworkStatsInputSpec(BaseInterfaceInputSpec):
    rsfc = traits.Dict(mandatory=True, dtype=float, desc='resting-state functional connectivity')

class _NetworkStatsOutputSpec(TraitedSpec):
    rs_stats = traits.Dict(dtype=float, desc='dynamic functional connectivity')

class NetworkStats(SimpleInterface):
    input_spec = _NetworkStatsInputSpec
    output_spec = _NetworkStatsOutputSpec

    def _run_interface(self, runtime):
        self._results['rs_stats'] = {'level1_strength': np.array([]), 'level1_betweenness': np.array([]),
                                     'level1_participation': np.array([]), 'level1_efficiency': np.array([]),
                                     'level2_strength': np.array([]), 'level2_betweenness': np.array([]),
                                     'level2_participation': np.array([]), 'level2_efficiency': np.array([]),
                                     'level3_strength': np.array([]), 'level3_betweenness': np.array([]),
                                     'level3_participation': np.array([]), 'level3_efficiency': np.array([]),
                                     'level4_strength': np.array([]), 'level4_betweenness': np.array([]),
                                     'level4_participation': np.array([]), 'level4_efficiency': np.array([])}

        for level in range(4):
            rsfc = self.inputs.rsfc[f'level{level+1}'].mean(axis=2)
            strength = bct.strengths_und(rsfc)
            betweenness = bct.betweenness_wei(rsfc)
            participation = bct.participation_coef(rsfc, bct.community_louvain(rsfc, B='negative_sym')[0])
            efficiency = bct.efficiency_wei(rsfc, local=True)

            self._results['rs_stats'][f'level{level+1}_strength'] = strength
            self._results['rs_stats'][f'level{level+1}_betweenness'] = betweenness
            self._results['rs_stats'][f'level{level+1}_participation'] = participation
            self._results['rs_stats'][f'level{level+1}_efficiency'] = efficiency

        return runtime

### TFC: compute task-based functional connectivity

class _TFCInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    t_runs = traits.List(mandatory=True, desc='task run names')
    t_files = traits.Dict(mandatory=True, dtype=Path, desc='filenames of task fMRI data')

class _TFCOutputSpec(TraitedSpec):
    tfc = traits.Dict(dtype=dict, desc='task-based functional connectivity')

class TFC(SimpleInterface):
    input_spec = _TFCInputSpec
    output_spec = _TFCOutputSpec

    def _run_interface(self, runtime):
        self._results['tfc'] = {'level1': np.array([]), 'level2': np.array([]),
                                'level3': np.array([]), 'level4': np.array([])}

        for run in self.inputs.t_runs:
            if self.inputs.t_files[f'{run}_surf'] and self.inputs.t_files[f'{run}_vol']:
                t_surf = nib.load(self.inputs.t_files[f'{run}_surf']).get_fdata()
                t_vol = nib.load(self.inputs.t_files[f'{run}_vol']).get_fdata()          
                self._results['tfc'], _ = fc(t_surf, t_vol, self.inputs.dataset, self.inputs.t_files, 
                                             self._results['tfc'])

        return runtime

### MyelinEstimate: extract the myelin estimates from T1dividedbyT2 files

class _MyelineEstimateInputSpec(BaseInterfaceInputSpec):
    anat_files = traits.Dict(mandatory=True, dtype=Path, desc='filenames of anatomical data')

class _MyelineEstimateOutputSpec(TraitedSpec):
    myelin = traits.Dict(dtype=float, desc='myelin content estimates')

class MyelinEstimate(SimpleInterface):
    input_spec = _MyelineEstimateInputSpec
    output_spec = _MyelineEstimateOutputSpec

    def _run_interface(self, runtime):
        self._results['myelin'] = {'level1': np.array([]), 'level2': np.array([]),
                                   'level3': np.array([]), 'level4': np.array([])}

        myelin_l = nib.load(self.inputs.anat_files['myelin_l']).agg_data()
        myelin_r = nib.load(self.inputs.anat_files['myelin_r']).agg_data()
        myelin_surf = np.hstack((myelin_l, myelin_r))
        myelin_vol = nib.load(self.inputs.anat_files['myelin_vol']).get_fdata()

        for level in range(4):
            parc_Sch_file = Path(base_dir, 'data', 'atlas', 
                                 f'Schaefer2018_{level+1}00Parcels_17Networks_order.dlabel.nii')
            parc_Sch = nib.load(parc_Sch_file).get_fdata()
            parc_Mel_file = Path(base_dir, 'data', 'atlas', f'Tian_Subcortex_S{level+1}_3T.nii.gz')
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
            
            self._results['myelin'][f'level{level+1}'] = np.hstack([parc_surf, parc_vol])

        return runtime

### Morphometry: extract morphometry features

class _MorphometryInputSpec(BaseInterfaceInputSpec):
    anat_dir = traits.Str(mandatory=True, desc='absolute path to installed subject T1w directory')
    anat_files = traits.Dict(mandatory=True, dtype=Path, desc='filenames of anatomical data')
    subject = traits.Str(mandatory=True, desc='subject ID')

class _MorphometryOutputSpec(TraitedSpec):
    morph = traits.Dict(dtype=float, desc='morphometry features')

class Morphometry(SimpleInterface):
    input_spec = _MorphometryInputSpec
    output_spec = _MorphometryOutputSpec

    def _run_interface(self, runtime):
        self._results['morph'] = {'level1_GMV': np.array([]), 'level1_CS': np.array([]), 'level1_CT': np.array([]),
                                  'level2_GMV': np.array([]), 'level2_CS': np.array([]), 'level2_CT': np.array([]),
                                  'level3_GMV': np.array([]), 'level3_CS': np.array([]), 'level3_CT': np.array([]),
                                  'level4_GMV': np.array([]), 'level4_CS': np.array([]), 'level4_CT': np.array([])}

        for level in range(4): 
            stats_surf = pd.DataFrame()
            for hemi in ['lh', 'rh']:
                annot_file = Path(base_dir, 'data', 'label', 
                                  f'{hemi}.Schaefer2018_{level+1}00Parcels_17Networks_order.annot')
                hemi_table = f'{hemi}.fs_stats'
                subprocess.run(['mris_anatomical_stats', '-a', annot_file, '-noglobal', '-f', hemi_table, 
                        self.inputs.subject, hemi], env=dict(environ, **{'SUBJECTS_DIR': self.inputs.anat_dir}),
                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                hemi_stats = pd.read_table(hemi_table, header=0, skiprows=np.arange(51), delim_whitespace=True)
                hemi_stats.drop([0], inplace=True) # exclude medial wall
                stats_surf = stats_surf.append(hemi_stats)
            self._results['morph'][f'level{level+1}_CS'] = stats_surf['SurfArea'].values
            self._results['morph'][f'level{level+1}_CT'] = stats_surf['ThickAvg'].values

            seg_file = Path(base_dir, 'data', 'atlas', f'Tian_Subcortex_S{level+1}_3T.nii.gz')
            seg_up_file = f'S{level}_upsampled.nii.gz'
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

            self._results['morph'][f'level{level+1}_GMV'] = np.concatenate((stats_surf['GrayVol'].values, 
                                                            stats_vol['Volume_mm3'].values))
                                    
        return runtime

### GradientAC: extract gradient loadings and anatomical connectivity

class _GradientACInputSpec(BaseInterfaceInputSpec):
    features_dir = traits.Dict(mandatory=True, dtype=str, desc='absolute path to extracted features for each dataset')
    sublists = traits.Dict(mandatory=True, dtype=list, desc='list of subjects available in each dataset')
    cv_split = traits.Dict(mandatory=True, dtype=list, desc='list of subjects in the test split of each fold')
    cv_split_perm = traits.Dict(mandatory=True, dtype=list, desc='list of permuted subjects')
    repeat = traits.Int(mandatory=True, desc='current repeat of cross-validation')
    fold = traits.Int(mandatory=True, desc='current fold in the repeat')
    level = traits.Str(mandatory=True, desc='parcellation level')
    config = traits.Dict({}, desc='configuration settings')

class _GradientACOutputSpec(TraitedSpec):
    embeddings = traits.Dict(desc='embeddings for gradients')
    params = traits.Dict(desc='parameters for anatomical connectivity')
    repeat = traits.Int(desc='current repeat of cross-validation')
    fold = traits.Int(desc='current fold in the repeat')
    level = traits.Str(desc='parcellation level')

class GradientAC(SimpleInterface):
    input_spec = _GradientACInputSpec
    output_spec = _GradientACOutputSpec

    def _extract_features(self):
        all_sub = sum(self.inputs.sublists.values(), [])
        image_features = dict.fromkeys(all_sub)
        
        for subject in all_sub:
            dataset = [key for key in self.inputs.sublists if subject in self.inputs.sublists[key]][0]
            if dataset == 'HCP-A' or 'HCP-D':
                feature_file = Path(self.inputs.features_dir[dataset], f'{dataset}_{subject}_V1_MR.h5')
            else:
                feature_file = Path(self.inputs.features_dir[dataset], f'{dataset}_{subject}.h5')

            feature_sub = {}
            feature_sub['rsfc'] = read_h5(feature_file, f'/rsfc/level{self.inputs.level}')
            feature_sub['myelin'] = read_h5(feature_file, f'/myelin/level{self.inputs.level}')
            for stats in ['GMV', 'CS', 'CT']:
                ds_morph = f'/morphometry/{stats}/level{self.inputs.level}'
                feature_sub[stats] = read_h5(feature_file, ds_morph)  
            image_features[subject] = feature_sub
        
        return image_features

    def _compute_features(self, image_features, cv_split, repeat):
        all_sub = sum(self.inputs.sublists.values(), [])
        gradients = dict.fromkeys(all_sub)
        ac = dict.fromkeys(all_sub)
        n_folds = int(self.inputs.config['n_folds'])

        test_sub = cv_split[f'repeat{repeat}_fold{self.inputs.fold}']
        val_sub = cv_split[f'repeat{repeat}_fold{(self.inputs.fold+1)%n_folds}']
        testval_sub = np.concatenate((val_sub, test_sub))
        train_sub = [subject for subject in all_sub if subject not in testval_sub]

        _, embed = diffusion_mapping(image_features, train_sub, 'rsfc', gradients)
        for feature in ['GMV', 'CS', 'CT', 'myelin']:
            _, params = score(image_features, train_sub, feature, ac)
            
        return embed, params
    
    def _run_interface(self, runtime):
        image_features = self._extract_features()
        self._results['embeddings'] = {}
        self._results['params'] = {}

        embed, params = self._compute_features(image_features, self.inputs.cv_split, self.inputs.repeat)
        self._results['embeddings']['embedding'] = embed
        self._results['params']['params'] = params

        # assuming n_repeats_perm = n_repeats x 10
        if int(self.inputs.config['n_repeats_perm']) == (int(self.inputs.config['n_repeats']) * 10):
            for repeat_split in range(10):
                repeat = int(self.inputs.repeat) * 10 + repeat_split
                embed, params = self._compute_features(image_features, self.inputs.cv_split_perm, repeat)
                self._results['embeddings'][f'repeat{repeat}'] = embed
                self._results['params'][f'repeat{repeat}'] = params

        self._results['repeat'] = self.inputs.repeat
        self._results['fold'] = self.inputs.fold
        self._results['level'] = self.inputs.level

        return runtime
