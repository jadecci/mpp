from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, File, traits
import datalad.api as dl
from os import path
from pathlib import Path
import logging

from mpp.utilities.data import write_h5

base_dir = path.join(path.dirname(path.realpath(__file__)), '..')
logging.getLogger('datalad').setLevel(logging.WARNING)

### InitData: install and get dataset

class _InitDataInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    work_dir = traits.Str(desc='absolute path to work directory')

class _InitDataOutputSpec(TraitedSpec):
    dataset_dir = traits.Str(desc='absolute path to installed dataset directory')

class InitData(SimpleInterface):
    input_spec = _InitDataInputSpec
    output_spec = _InitDataOutputSpec

    def _run_interface(self, runtime):
        base_dataset_dir = path.join(self.inputs.work_dir, self.inputs.dataset)
        dataset_url = {'HCP-YA': 'git@github.com:datalad-datasets/human-connectome-project-openaccess.git',
                       'HCP-A': 'git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git',
                       'HCP-D': 'git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git',
                       'ABCD': 'git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git'}
        dataset_dir = {'HCP-YA': path.join(base_dataset_dir, 'HCP1200'),
                       'HCP-A': path.join(base_dataset_dir, 'original', 'hcp', 'hcp_aging'),
                       'HCP-D': path.join(base_dataset_dir, 'original', 'hcp', 'hcp_development'),
                       'ABCD': path.join(base_dataset_dir, 'original', 'abcd', 'abcd-hcp-pipeline')}
        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            source = 'inm7-storage'
        else:
            source = None
        
        dl.install(path=base_dataset_dir, source=dataset_url[self.inputs.dataset], on_failure='stop')
        dl.get(path=dataset_dir[self.inputs.dataset], dataset=base_dataset_dir, get_data=False, source=source,
               on_failure='stop')  

        self._results['dataset_dir'] = dataset_dir[self.inputs.dataset]

        return runtime

### InitSubData: install and get subject-specific subdataset

class _InitSubDataInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    dataset_dir = traits.Str(desc='absolute path to installed dataset directory')
    subject = traits.Str(desc='subject ID')

class _InitSubDataOutputSpec(TraitedSpec):
    rs_dir = traits.Str(desc='absolute path to installed subject MNINonLinear directory')
    anat_dir = traits.Str(desc='absolute path to installed subject T1w directory')
    rs_runs = traits.List(desc='resting-state run names')
    t_runs = traits.List([], usedefault=True, desc='task run names')
    rs_files = traits.Dict(dtype=str, desc='filenames of resting-state data')
    t_files = traits.Dict({}, dtype=str, usedefault=True, desc='filenames of task fMRI data')
    anat_files = traits.Dict(dtype=str, desc='filenames of anatomical data')
    hcpd_b_runs = traits.Int(desc='number of HCP-D b runs')

class InitSubData(SimpleInterface):
    input_spec = _InitSubDataInputSpec
    output_spec = _InitSubDataOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            source = 'inm7-storage'
        else:
            source = None

        subject_dir = path.join(self.inputs.dataset_dir, self.inputs.subject)
        dl.get(path=subject_dir, dataset=self.inputs.dataset_dir, get_data=False, source=source, on_failure='stop')

        if 'HCP' in self.inputs.dataset:
            move_file = {'HCP-YA': 'Movement_Regressors.txt', 'HCP-A': 'Movement_Regressors_hp0_clean.txt',
                             'HCP-D': 'Movement_Regressors_hp0_clean.txt'}

            self._results['rs_dir'] = path.join(subject_dir, 'MNINonLinear')
            dl.get(path=self._results['rs_dir'], dataset=self.inputs.dataset_dir, get_data=False, source=source, 
                   on_failure='stop')
            anat_dir = path.join(subject_dir, 'T1w')
            dl.get(path=anat_dir, dataset=self.inputs.dataset_dir, get_data=False, source=source, on_failure='stop')

            # check rfMRI data
            runs = {'HCP-YA': ['rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR', 'rfMRI_REST2_RL'],
                    'HCP-A': ['rfMRI_REST1_AP', 'rfMRI_REST1_PA', 'rfMRI_REST2_AP', 'rfMRI_REST2_PA'],
                    'HCP-D': ['rfMRI_REST1_AP', 'rfMRI_REST1_PA', 'rfMRI_REST2_AP', 'rfMRI_REST2_PA']}
            self._results['rs_runs'] = runs[self.inputs.dataset]
            self._results['rs_files'] = {'wm_mask': path.join(subject_dir, 'MNINonLinear', 'ROIs', 
                                                              'Atlas_wmparc.2.nii.gz')}

            for i in range(4):
                run = runs[self.inputs.dataset][i]
                run_dir = path.join(subject_dir, 'MNINonLinear', 'Results', run)
                self._results['rs_files'][f'{run}_surf'] = path.join(run_dir, 
                                                                     f'{run}_Atlas_MSMAll_hp0_clean.dtseries.nii')
                self._results['rs_files'][f'{run}_vol'] = path.join(run_dir, f'{run}_hp0_clean.nii.gz')
                self._results['rs_files'][f'{run}_movement'] = path.join(run_dir, move_file[self.inputs.dataset])
            
            self._results['hcpd_b_runs'] = 0
            for key in self._results['rs_files']:
                if not path.islink(self._results['rs_files'][key]):
                    if self.inputs.dataset == 'HCP-D':
                        if 'AP' in self._results['rs_files'][key]:
                            rs_file_a = self._results['rs_files'][key].replace('_AP', 'a_AP')
                            rs_file_b = self._results['rs_files'][key].replace('_AP', 'b_AP')
                        elif 'PA' in self._results['rs_files'][key]:
                            rs_file_a = self._results['rs_files'][key].replace('_PA', 'a_PA')
                            rs_file_b = self._results['rs_files'][key].replace('_PA', 'b_PA')
                        self._results['rs_files'][key] = ''
                        if path.islink(rs_file_a):
                            self._results['rs_files'][key] = rs_file_a
                        if path.islink(rs_file_b):
                            self._results['rs_files'][f'{key}b'] = rs_file_b
                            self._results['hcpd_b_runs'] = self._results['hcpd_b_runs'] + 1
                    else:
                        self._results['rs_files'][key] = ''

            # check task data
            if 'HCP' in self.inputs.dataset:
                runs = {'HCP-YA': ['tfMRI_EMOTION_LR', 'tfMRI_EMOTION_RL', 'tfMRI_GAMBLING_LR', 'tfMRI_GAMBLING_RL',
                                   'tfMRI_LANGUAGE_LR', 'tfMRI_LANGUAGE_RL', 'tfMRI_MOTOR_LR', 'tfMRI_MOTOR_RL',
                                   'tfMRI_RELATIONAL_LR', 'tfMRI_RELATIONAL_RL', 'tfMRI_SOCIAL_LR', 'tfMRI_SOCIAL_RL',
                                   'tfMRI_WM_LR', 'tfMRI_WM_RL'],
                        'HCP-A': ['tfMRI_CARIT_PA', 'tfMRI_FACENAME_PA', 'tfMRI_VISMOTOR_PA'],
                        'HCP-D': ['tfMRI_CARIT_AP', 'tfMRI_CARIT_PA', 'tfMRI_EMOTION_PA', 'tfMRI_GUESSING_AP',
                                  'tfMRI_GUESSING_PA']}
                self._results['t_runs'] = runs[self.inputs.dataset]
                self._results['t_files'] = {'wm_mask': path.join(subject_dir, 'MNINonLinear', 'ROIs', 
                                                                'Atlas_wmparc.2.nii.gz')}

                for run in runs[self.inputs.dataset]:
                    run_dir = path.join(subject_dir, 'MNINonLinear', 'Results', run)
                    if self.inputs.dataset == 'HCP-YA':
                        self._results['t_files'][f'{run}_surf'] = path.join(run_dir, f'{run}_Atlas_MSMAll.dtseries.nii')
                        self._results['t_files'][f'{run}_vol'] = path.join(run_dir, f'{run}.nii.gz')
                    elif self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
                        self._results['t_files'][f'{run}_surf'] = path.join(run_dir, 
                                                                  f'{run}_Atlas_MSMAlll_hp0_clean.dtseries.nii')
                        self._results['t_files'][f'{run}_vol'] = path.join(run_dir, f'{run}_hp0_clean.nii.gz')
                    self._results['t_files'][f'{run}_movement'] = path.join(run_dir,  move_file[self.inputs.dataset])

                for key in self._results['t_files']:
                    if not path.islink(self._results['t_files'][key]):
                        self._results['t_files'][key] = ''

            # check sMRI data
            anat_dir = path.join(subject_dir, 'T1w', self.inputs.subject)
            self._results['anat_dir'] = path.join(subject_dir, 'T1w')
            self._results['anat_files'] = {'t1_vol': path.join(subject_dir, 'MNINonLinear', 'T1w.nii.gz'),
                                           'myelin_l': path.join(subject_dir, 'MNINonLinear', 'fsaverage_LR32k', 
                                                       f'{self.inputs.subject}.L.MyelinMap.32k_fs_LR.func.gii'),
                                           'myelin_r': path.join(subject_dir, 'MNINonLinear', 'fsaverage_LR32k', 
                                                       f'{self.inputs.subject}.R.MyelinMap.32k_fs_LR.func.gii'),
                                           'wm_vol': path.join(anat_dir, 'mri', 'wm.mgz'),
                                           'white_l': path.join(anat_dir, 'surf', 'lh.white'),
                                           'white_r': path.join(anat_dir, 'surf', 'rh.white'),
                                           'pial_l': path.join(anat_dir, 'surf', 'lh.pial'),
                                           'pial_r': path.join(anat_dir, 'surf', 'rh.pial'),
                                           'ct_l': path.join(anat_dir, 'surf', 'lh.thickness'),
                                           'ct_r': path.join(anat_dir, 'surf', 'rh.thickness'),
                                           'label_l': path.join(anat_dir, 'label', 'lh.cortex.label'),
                                           'label_r': path.join(anat_dir, 'label', 'rh.cortex.label'),
                                           'myelin_vol': path.join(subject_dir, 'T1w', 'T1wDividedByT2w.nii.gz')}
            for key in self._results['anat_files']:
                if not path.islink(self._results['anat_files'][key]):
                    self._results['anat_files'][key] = ''

        return runtime

### InitfMRIData: download fMRI data

class _InitfMRIDataInputSpec(BaseInterfaceInputSpec):
    func_dir = traits.Str(desc='absolute path to installed subject MNINonLinear directory')
    func_files = traits.Dict(dtype=str, desc='filenames of resting-state data')
    dataset = traits.Str(desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')

class _InitfMRIDataOutputSpec(TraitedSpec):
    func_files = traits.Dict(dtype=str, desc='filenames of resting-state data')

class InitfMRIData(SimpleInterface):
    input_spec = _InitfMRIDataInputSpec
    output_spec = _InitfMRIDataOutputSpec

    def _run_interface(self, runtime):
        for key in self.inputs.func_files:
            if self.inputs.func_files[key] != '':
                if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
                    source = 'inm7-storage'
                else:
                    source = None
                dl.get(path=self.inputs.func_files[key], dataset=self.inputs.func_dir, source=source, on_failure='stop')
        
        self._results['func_files'] = self.inputs.func_files

        return runtime

### InitAnatData: download anatomical data

class _InitAnatDataInputSpec(BaseInterfaceInputSpec):
    rs_dir = traits.Str(desc='absolute path to installed subject MNINonLinear directory')
    anat_dir = traits.Str(desc='absolute path to installed subject T1w directory')
    anat_files = traits.Dict(dtype=str, desc='filenames of anatomical data')
    dataset = traits.Str(desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')

class _InitAnatDataOutputSpec(TraitedSpec):
    anat_files = traits.Dict(dtype=str, desc='filenames of anatomical data')

class InitAnatData(SimpleInterface):
    input_spec = _InitAnatDataInputSpec
    output_spec = _InitAnatDataOutputSpec

    def _run_interface(self, runtime):
        for key in self.inputs.anat_files:
            if self.inputs.anat_files[key] != '':
                if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
                    source = 'inm7-storage'
                else:
                    source = None

                if key == 't1_vol' or key == 'myelin_l' or key == 'myelin_r':
                    dl.get(path=self.inputs.anat_files[key], dataset=self.inputs.rs_dir, source=source, 
                            on_failure='stop')
                else:
                    dl.get(path=self.inputs.anat_files[key], dataset=self.inputs.anat_dir, source=source, 
                            on_failure='stop')

        self._results['anat_files'] = self.inputs.anat_files

        return runtime

### SaveFeatures: save extracted features
class _SaveFeaturesInputSpec(BaseInterfaceInputSpec):
    rsfc = traits.Dict(dtype=float, desc='resting-state functional connectivity')
    dfc = traits.Dict(dtype=float, desc='dynamic functional connectivity')
    rs_stats = traits.Dict(dtype=float, desc='dynamic functional connectivity')
    tfc = traits.Dict({}, usedefault=True, dtype=dict, desc='task-based functional connectivity')
    myelin = traits.Dict(dtype=float, desc='myelin content estimates')
    morph = traits.Dict(dtype=float, desc='morphometry features')
    output_dir = traits.Str(desc='absolute path to output directory')
    dataset = traits.Str(desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    subject = traits.Str(desc='subject ID')
    overwrite = traits.Bool(desc='whether to overwrite existing results')

class _SaveFeaturesOutputSpec(TraitedSpec):
    sub_done = traits.Bool(False, usedefault=True, desc='whether subject workflow is completed')

class SaveFeatures(SimpleInterface):
    input_spec = _SaveFeaturesInputSpec
    output_spec = _SaveFeaturesOutputSpec

    def _run_interface(self, runtime):
        Path(self.inputs.output_dir).mkdir(parents=True, exist_ok=True)
        output_file = path.join(self.inputs.output_dir, f'{self.inputs.dataset}_{self.inputs.subject}.h5')

        for level in range(4):
            ds_rsfc = f'/rsfc/level{level+1}'
            data_rsfc = self.inputs.rsfc[f'level{level+1}']
            write_h5(output_file, ds_rsfc, data_rsfc, self.inputs.overwrite)

            ds_dfc = f'/dfc/level{level+1}'
            data_dfc = self.inputs.dfc[f'level{level+1}']
            write_h5(output_file, ds_dfc, data_dfc, self.inputs.overwrite)

            for stats in ['strength', 'betweenness', 'participation', 'efficiency']:
                ds_stats = f'/network_stats/{stats}/level{level+1}'
                data_stats = self.inputs.rs_stats[f'level{level+1}_{stats}']
                write_h5(output_file, ds_stats, data_stats, self.inputs.overwrite)
    
            if self.inputs.tfc:
                for key in self.inputs.tfc:
                    ds_tfc = f'/tfc/{key}/level{level+1}'
                    data_tfc = self.inputs.tfc[key][f'level{level+1}']
                    write_h5(output_file, ds_tfc, data_tfc, self.inputs.overwrite)

            for stats in ['GMV', 'CS', 'CT']:
                ds_morph = f'/morphometry/{stats}/level{level+1}'
                data_morph = self.inputs.morph[f'level{level+1}_{stats}']
                write_h5(output_file, ds_morph, data_morph, self.inputs.overwrite)

            ds_myelin = f'/myelin/level{level+1}'
            data_myelin = self.inputs.myelin[f'level{level+1}']
            write_h5(output_file, ds_myelin, data_myelin, self.inputs.overwrite)

        self._results['sub_done'] = True

        return runtime

### DropSubjectData: drop subject-wise data after feature extraction is done

class _DropSubDataInputSpec(BaseInterfaceInputSpec):
    rs_dir = traits.Str(desc='absolute path to installed subject MNINonLinear directory')  
    rs_files = traits.Dict(dtype=str, desc='filenames of resting-state data')
    t_files = traits.Dict(dtype=str, desc='filenames of task fMRI data')
    anat_dir = traits.Str(desc='absolute path to installed subject T1w directory')
    anat_files = traits.Dict(dtype=str, desc='filenames of anatomical data')
    sub_done = traits.Bool(False, usedefault=True, desc='whether subject workflow is completed')

class DropSubData(SimpleInterface):
    input_spec = _DropSubDataInputSpec

    def _run_interface(self, runtime):
        if self.inputs.sub_done:
            for key in self.inputs.rs_files:
                if self.inputs.rs_files[key] != '':
                    dl.drop(path=self.inputs.rs_files[key], dataset=self.inputs.rs_dir, on_failure='stop')
            
            for key in self.inputs.t_files:
                if self.inputs.t_files[key] != '':
                    dl.drop(path=self.inputs.t_files[key], dataset=self.inputs.rs_dir, on_failure='stop')

            for key in self.inputs.anat_files:
                if self.inputs.anat_files[key] != '':
                    if key == 't1_vol' or key == 'myelin_l' or key == 'myelin_r':
                        dl.drop(path=self.inputs.anat_files[key], dataset=self.inputs.rs_dir, on_failure='stop')
                    else:
                        dl.drop(path=self.inputs.anat_files[key], dataset=self.inputs.anat_dir, on_failure='stop')

        return runtime