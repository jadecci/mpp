from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, File, traits
import datalad.api as dl
from os import path
import logging

from mpp import logger
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
        
        logger.info(f'Installing {self.inputs.dataset} dataset into {base_dataset_dir}')
        dl.install(path=base_dataset_dir, source=dataset_url[self.inputs.dataset], on_failure='stop')

        logger.info(f'Getting {dataset_dir[self.inputs.dataset]} from dataset at {base_dataset_dir}')
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
    rs_files = traits.Dict(desc='filenames of resting-state data')
    anat_files = traits.Dict(desc='filenames of anatomical data')
    rs_skip = traits.Bool(desc='whether resting-state workflow should be skipped or not')
    t1_skip = traits.Bool(desc='whether morphometry feature computation should be skipped or not')
    myelin_skip = traits.Bool(desc='whether myelin feature computation should be skipped or not')

class InitSubData(SimpleInterface):
    input_spec = _InitSubDataInputSpec
    output_spec = _InitSubDataOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            source = 'inm7-storage'
        else:
            source = None

        subject_dir = path.join(self.inputs.dataset_dir, self.inputs.subject)
        logger.info(f'Getting {subject_dir} from dataset {self.inputs.dataset_dir}')
        dl.get(path=subject_dir, dataset=self.inputs.dataset_dir, get_data=False, source=source, on_failure='stop')

        if 'HCP' in self.inputs.dataset:
            rs_dir = path.join(subject_dir, 'MNINonLinear')
            logger.info(f'Getting {rs_dir} from dataset {self.inputs.dataset_dir}')
            dl.get(path=rs_dir, dataset=self.inputs.dataset_dir, get_data=False, source=source, on_failure='stop')

            anat_dir = path.join(subject_dir, 'T1w')
            logger.info(f'Getting {anat_dir} from dataset {self.inputs.dataset_dir}')
            dl.get(path=anat_dir, dataset=self.inputs.dataset_dir, get_data=False, source=source, on_failure='stop')

            # check rfMRI data
            runs = {'HCP-YA': ['rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR', 'rfMRI_REST2_RL'],
                    'HCP-A': ['rfMRI_REST1_AP', 'rfMRI_REST1_PA', 'rfMRI_REST2_AP', 'rfMRI_REST2_PA'],
                    'HCP-D': ['rfMRI_REST1_AP', 'rfMRI_REST1_PA', 'rfMRI_REST2_AP', 'rfMRI_REST2_PA']}
            movement_file = {'HCP-YA': 'Movement_Regressors.txt', 'HCP-A': 'Movement_Regressors_hp0_clean.txt',
                             'HCP-D': 'Movement_Regressors_hp0_clean.txt'}
            rs_files = {'run1_surf': path.join(subject_dir, 'MNINonLinear', 'Results', runs[self.inputs.dataset][0],
                                              (runs[self.inputs.dataset][0] + '_Atlas_MSMAll_hp0_clean.dtseries.nii')),
                        'run2_surf': path.join(subject_dir, 'MNINonLinear',  'Results', runs[self.inputs.dataset][1],
                                              (runs[self.inputs.dataset][1] + '_Atlas_MSMAll_hp0_clean.dtseries.nii')),
                        'run3_surf': path.join(subject_dir, 'MNINonLinear',  'Results', runs[self.inputs.dataset][2],
                                              (runs[self.inputs.dataset][2] + '_Atlas_MSMAll_hp0_clean.dtseries.nii')),
                        'run4_surf': path.join(subject_dir, 'MNINonLinear',  'Results', runs[self.inputs.dataset][3],
                                              (runs[self.inputs.dataset][3] + '_Atlas_MSMAll_hp0_clean.dtseries.nii')),
                        'run1_vol': path.join(subject_dir, 'MNINonLinear',  'Results', runs[self.inputs.dataset][0],
                                              (runs[self.inputs.dataset][0] + '_hp0_clean.nii.gz')),
                        'run2_vol': path.join(subject_dir, 'MNINonLinear',  'Results', runs[self.inputs.dataset][1],
                                              (runs[self.inputs.dataset][1] + '_hp0_clean.nii.gz')),
                        'run3_vol': path.join(subject_dir, 'MNINonLinear',  'Results', runs[self.inputs.dataset][2],
                                              (runs[self.inputs.dataset][2] + '_hp0_clean.nii.gz')),
                        'run4_vol': path.join(subject_dir, 'MNINonLinear',  'Results', runs[self.inputs.dataset][3],
                                              (runs[self.inputs.dataset][3] + '_hp0_clean.nii.gz')),
                        'wm_mask': path.join(subject_dir, 'MNINonLinear', 'ROIs', 'Atlas_wmparc.2.nii.gz'),
                        'run1_movement': path.join(subject_dir, 'MNINonLinear', 'Results', runs[self.inputs.dataset][0],
                                                   movement_file[self.inputs.dataset]),
                        'run2_movement': path.join(subject_dir, 'MNINonLinear', 'Results', runs[self.inputs.dataset][1],
                                                   movement_file[self.inputs.dataset]),
                        'run3_movement': path.join(subject_dir, 'MNINonLinear', 'Results', runs[self.inputs.dataset][2],
                                                   movement_file[self.inputs.dataset]),
                        'run4_movement': path.join(subject_dir, 'MNINonLinear', 'Results', runs[self.inputs.dataset][3],
                                                   movement_file[self.inputs.dataset])}
            for key in rs_files:
                if not path.islink(rs_files[key]):
                    if self.inputs.dataset == 'HCP-A':
                        if 'AP' in rs_files[key]:
                            rs_file_alter = rs_files[key].replace('_AP', 'a_AP')
                        elif 'PA' in rs_files[key]:
                            rs_file_alter = rs_files[key].replace('_PA', 'a_PA')
                        rs_files[key] = ''
                        if path.islink(rs_file_alter):
                            rs_files[key] = rs_file_alter
                    else:
                        rs_files[key] = ''

            if (rs_files['run1_surf'] =='' and rs_files['run2_surf'] =='' and rs_files['run3_surf'] =='' and 
                rs_files['run4_surf'] ==''):
                rs_surf_skip = True
            else:
                rs_surf_skip = False

            if (rs_files['run1_vol'] =='' and rs_files['run2_vol'] =='' and rs_files['run3_vol'] =='' and 
                rs_files['run4_vol'] ==''):
                rs_vol_skip = True
            else:
                rs_vol_skip = False

            self._results['rs_skip'] = rs_surf_skip or rs_vol_skip

            # check sMRI data
            anat_dir = path.join(subject_dir, 'T1w', self.inputs.subject)
            anat_files = {'t1_vol': path.join(subject_dir, 'MNINonLinear', 'T1w.nii.gz'),
                          'myelin_l': path.join(subject_dir, 'MNINonLinear', 'fsaverage_LR32k', 
                                                    (self.inputs.subject + '.L.MyelinMap.32k_fs_LR.func.gii')),
                          'myelin_r': path.join(subject_dir, 'MNINonLinear', 'fsaverage_LR32k', 
                                                    (self.inputs.subject + '.R.MyelinMap.32k_fs_LR.func.gii')),
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
            for key in anat_files:
                if not path.islink(anat_files[key]):
                    anat_files[key] = ''

            if (anat_files['t1_vol'] == '' or anat_files['wm_vol'] == '' or anat_files['white_l'] == '' or
                anat_files['white_r'] == '' or anat_files['pial_l'] == '' or anat_files['pial_r'] == '' or
                anat_files['ct_l'] == '' or anat_files['ct_r'] == '' or anat_files['label_l'] == '' or 
                anat_files['label_r'] == ''):
                self._results['t1_skip'] = True
            else:
                self._results['t1_skip'] = False

            if (anat_files['myelin_l'] == '' or anat_files['myelin_r'] == '' or anat_files['myelin_vol'] == ''):
                self._results['myelin_skip'] = True
            else:
                self._results['myelin_skip'] = False
            
        self._results['rs_dir'] = rs_dir
        self._results['anat_dir'] = path.join(subject_dir, 'T1w')
        self._results['rs_files'] = rs_files
        self._results['anat_files'] = anat_files

        return runtime

### InitRSData: download resting-state data

class _InitRSDataInputSpec(BaseInterfaceInputSpec):
    rs_dir = traits.Str(desc='absolute path to installed subject MNINonLinear directory')
    rs_files = traits.Dict(desc='filenames of resting-state data')
    rs_skip = traits.Bool(desc='whether resting-state workflow should be skipped or not')
    dataset = traits.Str(desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')

class _InitRSDataOutputSpec(TraitedSpec):
    rs_files = traits.Dict(desc='filenames of resting-state data')

class InitRSData(SimpleInterface):
    input_spec = _InitRSDataInputSpec
    output_spec = _InitRSDataOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.rs_skip:
            logger.warning('Resting-state workflow is skipped.')
        else:
            for key in self.inputs.rs_files:
                if self.inputs.rs_files[key] != '':
                    if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
                        source = 'inm7-storage'
                    else:
                        source = None

                    logger.info(f'Getting {self.inputs.rs_files[key]} from dataset {self.inputs.rs_dir}')
                    dl.get(path=self.inputs.rs_files[key], dataset=self.inputs.rs_dir, source=source, on_failure='stop')
            
            self._results['rs_files'] = self.inputs.rs_files

        return runtime

### InitAnatData: download anatomical data

class _InitAnatDataInputSpec(BaseInterfaceInputSpec):
    rs_dir = traits.Str(desc='absolute path to installed subject MNINonLinear directory')
    anat_dir = traits.Str(desc='absolute path to installed subject T1w directory')
    anat_files = traits.Dict(desc='filenames of anatomical data')
    t1_skip = traits.Bool(desc='whether morphometry feature computation should be skipped or not')
    myelin_skip = traits.Bool(desc='whether myelin feature computation should be skipped or not')
    dataset = traits.Str(desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')

class _InitAnatDataOutputSpec(TraitedSpec):
    anat_files = traits.Dict(desc='filenames of anatomical data')

class InitAnatData(SimpleInterface):
    input_spec = _InitAnatDataInputSpec
    output_spec = _InitAnatDataOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.t1_skip and self.inputs.myelin_skip:
            logger.warning('Anatomical workflow is skipped.')
        else:
            for key in self.inputs.anat_files:
                if self.inputs.anat_files[key] != '':
                    if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
                        source = 'inm7-storage'
                    else:
                        source = None

                    if key == 't1_vol' or key == 'myelin_l' or key == 'myelin_r':
                        logger.info(f'Getting {self.inputs.anat_files[key]} from dataset {self.inputs.rs_dir}')
                        dl.get(path=self.inputs.anat_files[key], dataset=self.inputs.rs_dir, source=source, 
                               on_failure='stop')
                    else:
                        logger.info(f'Getting {self.inputs.anat_files[key]} from dataset {self.inputs.anat_dir}')
                        dl.get(path=self.inputs.anat_files[key], dataset=self.inputs.anat_dir, source=source, 
                               on_failure='stop')

            self._results['anat_files'] = self.inputs.anat_files

        return runtime

### RSSave: save resting-state features to file

class _RSSaveInputSpec(BaseInterfaceInputSpec):
    rsfc = traits.Dict(dtype=float, desc='resting-state functional connectivity')
    dfc = traits.Dict(dtype=float, desc='dynamic functional connectivity')
    rs_stats = traits.Dict(dtype=float, desc='dynamic functional connectivity')
    output_dir = traits.Str(desc='absolute path to output directory')
    dataset = traits.Str(desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    subject = traits.Str(desc='subject ID')
    overwrite = traits.Bool(desc='whether to overwrite existing results')
    rs_skip = traits.Bool(desc='whether resting-state workflow should be skipped or not')

class _RSSaveOutputSpec(TraitedSpec):
    rs_done = traits.Bool(False, usedefault=True, desc='whether resting-state workflow is completed')

class RSSave(SimpleInterface):
    input_spec = _RSSaveInputSpec
    output_spec = _RSSaveOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.rs_skip:
            logger.warning('Resting-state workflow is skipped.')
        else:
            rs_feature_file = path.join(self.inputs.output_dir, (self.inputs.dataset + '_' + self.inputs.subject +
                                                                 '_rs_features.h5'))
            logger.info(f'Saving resting-state features to {rs_feature_file}')

            for level in range(4):
                ds_rsfc = '/rsfc/level' + str(level+1)
                data_rsfc = self.inputs.rsfc[('level' + str(level+1))]
                write_h5(rs_feature_file, ds_rsfc, data_rsfc, self.inputs.overwrite)

                ds_dfc = '/dfc/level' + str(level+1)
                data_dfc = self.inputs.dfc[('level' + str(level+1))]
                write_h5(rs_feature_file, ds_dfc, data_dfc, self.inputs.overwrite)

                for stats in ['strength', 'betweenness', 'participation', 'efficiency']:
                    ds_stats = '/network_stats/' + stats + '/level' + str(level+1)
                    data_stats = self.inputs.rs_stats[('level' + str(level+1) + '_' + stats)]
                    write_h5(rs_feature_file, ds_stats, data_stats, self.inputs.overwrite)

            self._results['rs_done'] = True

        return runtime

### AnatSave: save anatomical features to file

class _AnatSaveInputSpec(BaseInterfaceInputSpec):
    myelin = traits.Dict(desc='myelin content estimates')
    morph = traits.Dict(desc='morphometry features')
    output_dir = traits.Str(desc='absolute path to output directory')
    dataset = traits.Str(desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    subject = traits.Str(desc='subject ID')
    overwrite = traits.Bool(desc='whether to overwrite existing results')
    t1_skip = traits.Bool(desc='whether morphometry feature computation should be skipped or not')
    myelin_skip = traits.Bool(desc='whether myelin feature computation should be skipped or not')

class _AnatSaveOutputSpec(TraitedSpec):
    anat_done = traits.Bool(False, usedefault=True, desc='whether anatomical workflow is completed')

class AnatSave(SimpleInterface):
    input_spec = _AnatSaveInputSpec
    output_spec = _AnatSaveOutputSpec

    def _run_interface(self, runtime):
        anat_feature_file = path.join(self.inputs.output_dir, (self.inputs.dataset + '_' + self.inputs.subject + 
                                                               '_anat_features.h5'))
        logger.info(f'Saving anatomical features to {anat_feature_file}')

        if self.inputs.t1_skip:
            logger.warning('Morphometry features are skipped.')
        else:
            for level in range(4):
                for stats in ['GMV', 'CS', 'CT']:
                    ds_morph = '/morphometry/' + stats + '/level' + str(level+1)
                    data_morph = self.inputs.morph[('level' + str(level+1) + '_' + stats)]
                    write_h5(anat_feature_file, ds_morph, data_morph, self.inputs.overwrite)

            self._results['anat_done'] = True

        if self.inputs.myelin_skip:
            logger.warning('Myelin features are skipped.')
        else:
            for level in range(4):
                ds_myelin = '/myelin/level' + str(level+1)
                data_myelin = self.inputs.myelin[('level' + str(level+1))]
                write_h5(anat_feature_file, ds_myelin, data_myelin, self.inputs.overwrite)

            self._results['anat_done'] = True

        return runtime

### DropSubjectData: drop subject-wise data after feature extraction is done

class _DropSubDataInputSpec(BaseInterfaceInputSpec):
    rs_dir = traits.Str(desc='absolute path to installed subject MNINonLinear directory')  
    rs_files = traits.Dict(desc='filenames of resting-state data')
    rs_done = traits.Bool(False, usedefault=True, desc='whether resting-state workflow is completed')
    anat_dir = traits.Str(desc='absolute path to installed subject T1w directory')
    anat_files = traits.Dict(desc='filenames of anatomical data')
    anat_done = traits.Bool(False, usedefault=True, desc='whether anatomical workflow is completed')

class DropSubData(SimpleInterface):
    input_spec = _DropSubDataInputSpec

    def _run_interface(self, runtime):
        if self.inputs.rs_done:
            for key in self.inputs.rs_files:
                    if self.inputs.rs_files[key] != '':
                        dl.drop(path=self.inputs.rs_files[key], dataset=self.inputs.rs_dir, on_failure='stop')

        if self.inputs.anat_done:
            for key in self.inputs.anat_files:
                    if self.inputs.anat_files[key] != '':
                        if key == 't1_vol' or key == 'myelin_l' or key == 'myelin_r':
                            dl.drop(path=self.inputs.anat_files[key], dataset=self.inputs.rs_dir, on_failure='stop')
                        else:
                            dl.drop(path=self.inputs.anat_files[key], dataset=self.inputs.anat_dir, on_failure='stop')

        return runtime