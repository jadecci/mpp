import subprocess
from pathlib import Path
from os import getenv
import logging

import pandas as pd
import numpy as np
from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import datalad.api as dl

from mpp.utilities.data import write_h5, pheno_hcp
from mpp.utilities.features import pheno_conf_hcp
from mpp.exceptions import DatasetError

logging.getLogger('datalad').setLevel(logging.WARNING)

dataset_url = {
    'HCP-YA': 'git@github.com:datalad-datasets/human-connectome-project-openaccess.git',
    'HCP-A': 'git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git',
    'HCP-D': 'git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git',
    'ABCD': 'git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git'}

task_runs = {
    'HCP-YA': ['tfMRI_EMOTION_LR', 'tfMRI_EMOTION_RL', 'tfMRI_GAMBLING_LR', 'tfMRI_GAMBLING_RL',
               'tfMRI_LANGUAGE_LR', 'tfMRI_LANGUAGE_RL', 'tfMRI_MOTOR_LR', 'tfMRI_MOTOR_RL',
               'tfMRI_RELATIONAL_LR', 'tfMRI_RELATIONAL_RL', 'tfMRI_SOCIAL_LR', 'tfMRI_SOCIAL_RL',
               'tfMRI_WM_LR', 'tfMRI_WM_RL'],
    'HCP-A': ['tfMRI_CARIT_PA', 'tfMRI_FACENAME_PA', 'tfMRI_VISMOTOR_PA'],
    'HCP-D': ['tfMRI_CARIT_AP', 'tfMRI_CARIT_PA', 'tfMRI_EMOTION_PA', 'tfMRI_GUESSING_AP',
              'tfMRI_GUESSING_PA']}


class _InitDataInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(
        mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')
    subject = traits.Str(mandatory=True, desc='subject ID')
    output_dir = traits.Directory(mandatory=True, desc='absolute path to output directory')


class _InitDataOutputSpec(TraitedSpec):
    anat_dir = traits.Directory(desc='absolute path to installed subject T1w directory')
    rs_runs = traits.List(desc='resting-state run names')
    t_runs = traits.List(desc='task run names')
    rs_files = traits.Dict(dtype=Path, desc='filenames of resting-state data')
    t_files = traits.Dict(dtype=Path, desc='filenames of task fMRI data')
    anat_files = traits.Dict(dtype=Path, desc='filenames of anatomical data')
    hcpd_b_runs = traits.Int(desc='number of HCP-D b runs')
    hcpad_astats = traits.File(desc='aseg.stats file for HCP-A and HCP-D brain volume computation')
    dataset_dir = traits.Directory(desc='absolute path to installed root dataset')


class InitData(SimpleInterface):
    """Install and get subject-specific data"""
    input_spec = _InitDataInputSpec
    output_spec = _InitDataOutputSpec

    def _run_interface(self, runtime):
        self._results['dataset_dir'] = Path(self.inputs.work_dir, self.inputs.subject)
        dataset_dirs = {
            'HCP-YA': Path(self._results['dataset_dir'], 'HCP1200'),
            'HCP-A': Path(self._results['dataset_dir'], 'original', 'hcp', 'hcp_aging'),
            'HCP-D': Path(self._results['dataset_dir'], 'original', 'hcp', 'hcp_development'),
            'ABCD': Path(self._results['dataset_dir'], 'original', 'abcd', 'abcd-hcp-pipeline')}
        dataset_dir = dataset_dirs[self.inputs.dataset]

        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            source = 'inm7-storage'
        else:
            source = None

        # install datasets
        dl.install(
            path=self._results['dataset_dir'], source=dataset_url[self.inputs.dataset],
            on_failure='stop')
        dl.get(
            path=dataset_dir, dataset=self._results['dataset_dir'], get_data=False, source=source,
            on_failure='stop')
        subject_dir = Path(dataset_dir, self.inputs.subject)
        dl.get(
            path=subject_dir, dataset=dataset_dir, get_data=False, source=source, on_failure='stop')

        if self.inputs.dataset in ['HCP-YA', 'HCP-A', 'HCP-D']:
            rs_dir = Path(subject_dir, 'MNINonLinear')
            dl.get(
                path=rs_dir, dataset=dataset_dir, get_data=False, source=source, on_failure='stop')
            anat_dir = Path(subject_dir, 'T1w')
            dl.get(
                path=anat_dir, dataset=dataset_dir, get_data=False, source=source,
                on_failure='stop')

            # rfMRI data
            runs = {
                'HCP-YA': ['rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR', 'rfMRI_REST2_RL'],
                'HCP-A': ['rfMRI_REST1_AP', 'rfMRI_REST1_PA', 'rfMRI_REST2_AP', 'rfMRI_REST2_PA'],
                'HCP-D': ['rfMRI_REST1_AP', 'rfMRI_REST1_PA', 'rfMRI_REST2_AP', 'rfMRI_REST2_PA']}
            self._results['rs_runs'] = runs[self.inputs.dataset]
            rs_files = {
                'atlas_mask': Path(subject_dir, 'MNINonLinear', 'ROIs', 'Atlas_wmparc.2.nii.gz')}

            for i, run in enumerate(runs[self.inputs.dataset]):
                run = runs[self.inputs.dataset][i]
                run_dir = Path(subject_dir, 'MNINonLinear', 'Results', run)
                rs_files[f'{run}_surf'] = Path(
                    run_dir, f'{run}_Atlas_MSMAll_hp0_clean.dtseries.nii')
                rs_files[f'{run}_vol'] = Path(run_dir, f'{run}_hp0_clean.nii.gz')

            self._results['rs_files'] = rs_files.copy()
            self._results['hcpd_b_runs'] = 0
            for key, val in rs_files.items():
                if not val.is_symlink():
                    if self.inputs.dataset == 'HCP-D':
                        if 'AP' in val:
                            rs_file_a = Path(str(val).replace('_AP', 'a_AP'))
                            rs_file_b = Path(str(val).replace('_AP', 'b_AP'))
                        elif 'PA' in val:
                            rs_file_a = str(val).replace('_PA', 'a_PA')
                            rs_file_b = str(val).replace('_PA', 'b_PA')
                        else:
                            raise ValueError("file name %s has neither PA nor AP", val)
                        self._results['rs_files'][key] = ''
                        if rs_file_a.is_symlink:
                            self._results['rs_files'][key] = rs_file_a
                        if rs_file_b.is_symlink():
                            self._results['rs_files'][f'{key}b'] = rs_file_b
                            self._results['hcpd_b_runs'] = self._results['hcpd_b_runs'] + 1
                    else:
                        self._results['rs_files'][key] = ''

            # task data
            self._results['t_runs'] = task_runs[self.inputs.dataset]
            t_files = {
                'atlas_mask': Path(subject_dir, 'MNINonLinear', 'ROIs', 'Atlas_wmparc.2.nii.gz')}

            for run in task_runs[self.inputs.dataset]:
                run_dir = Path(subject_dir, 'MNINonLinear', 'Results', run)
                t_files[f'{run}_surf'] = Path(run_dir, f'{run}_Atlas_MSMAll.dtseries.nii')
                t_files[f'{run}_vol'] = Path(run_dir, f'{run}.nii.gz')
                # t_files[f'{run}_movement'] = path.join(run_dir,  move_file[self.inputs.dataset])
                # st_files[f'{run}_fd'] = path.join(run_dir, fd_file[self.inputs.dataset])

            self._results['t_files'] = t_files.copy()
            for key, val in t_files.items():
                if val.is_symlink():
                    dl.get(
                        path=self._results['t_files'][key], dataset=rs_dir, source=source,
                        on_failure='stop')
                else:
                    self._results['t_files'][key] = ''

            # sMRI data
            anat_dir = Path(subject_dir, 'T1w', self.inputs.subject)
            self._results['anat_dir'] = Path(subject_dir, 'T1w')
            anat_files = {
                't1_vol': Path(subject_dir, 'MNINonLinear', 'T1w.nii.gz'),
                'myelin_l': Path(
                    subject_dir, 'MNINonLinear', 'fsaverage_LR32k',
                    f'{self.inputs.subject}.L.MyelinMap.32k_fs_LR.func.gii'),
                'myelin_r': Path(
                    subject_dir, 'MNINonLinear', 'fsaverage_LR32k',
                    f'{self.inputs.subject}.R.MyelinMap.32k_fs_LR.func.gii'),
                'wm_vol': Path(anat_dir, 'mri', 'wm.mgz'),
                'white_l': Path(anat_dir, 'surf', 'lh.white'),
                'white_r': Path(anat_dir, 'surf', 'rh.white'),
                'pial_l': Path(anat_dir, 'surf', 'lh.pial'),
                'pial_r': Path(anat_dir, 'surf', 'rh.pial'),
                'ct_l': Path(anat_dir, 'surf', 'lh.thickness'),
                'ct_r': Path(anat_dir, 'surf', 'rh.thickness'),
                'label_l': Path(anat_dir, 'label', 'lh.cortex.label'),
                'label_r': Path(anat_dir, 'label', 'rh.cortex.label'),
                'myelin_vol': Path(subject_dir, 'T1w', 'T1wDividedByT2w.nii.gz')}

            self._results['anat_files'] = anat_files.copy()
            for key, val in anat_files.items():
                if val.is_symlink():
                    if key == 't1_vol' or key == 'myelin_l' or key == 'myelin_r':
                        dl.get(path=val, dataset=rs_dir, source=source, on_failure='stop')
                    else:
                        dl.get(
                            path=val, dataset=self._results['anat_dir'], source=source,
                            on_failure='stop')
                else:
                    self._results['anat_files'][key] = ''

            # get aseg stats table for HCP-A and HCP-D
            if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
                hcpad_astats = Path(anat_dir, 'stats', 'aseg.stats')
                output_dir = Path(self.inputs.output_dir, f'{self.inputs.dataset}_astats')
                astats_table = Path(output_dir, f'{self.inputs.subject}.txt')
                output_dir.mkdir(parents=True, exist_ok=True)

                dl.get(
                    path=hcpad_astats, dataset=self._results['anat_dir'], source=source,
                    on_failure='stop')
                subprocess.run(
                    ['python2', f'{getenv("FREESURFER_HOME")}/bin/asegstats2table', '--meas',
                     'volume', '--tablefile', astats_table, '--inputs', hcpad_astats], check=True)

        else:
            raise DatasetError()

        # get rfMRI data
        for val in self._results['rs_files'].values():
            if val != '':
                dl.get(path=val, dataset=rs_dir, source=source, on_failure='stop')

        return runtime


class _InitDiffusionDataInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(
        mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')
    subject = traits.Str(mandatory=True, desc='subject ID')
    output_dir = traits.Directory(mandatory=True, desc='absolute path to output directory')


class _InitDiffusionDataOutputSpec(TraitedSpec):
    d_files = traits.Dict(dtype=Path, desc='filenames of diffusion data')
    t1_files = traits.Dict(dtype=Path, desc='filenames of T1w data')
    fs_files = traits.Dict(dtype=Path, desc='filenames of FreeSurfer outputs')
    dataset_dir = traits.Directory(desc='absolute path to installed root dataset')
    fs_dir = traits.Directory(desc='FreeSurfer subject directory')


class InitDiffusionData(SimpleInterface):
    """Instal and get subject-specific diffusion data"""
    input_spec = _InitDiffusionDataInputSpec
    output_spec = _InitDiffusionDataOutputSpec

    def _run_interface(self, runtime):
        self._results['dataset_dir'] = Path(self.inputs.work_dir, self.inputs.subject)
        dataset_dirs = {
            'HCP-YA': Path(self._results['dataset_dir'], 'HCP1200'),
            'HCP-A': Path(self._results['dataset_dir'], 'original', 'hcp', 'hcp_aging'),
            'HCP-D': Path(self._results['dataset_dir'], 'original', 'hcp', 'hcp_development'),
            'ABCD': Path(self._results['dataset_dir'], 'original', 'abcd', 'abcd-hcp-pipeline')}
        dataset_dir = dataset_dirs[self.inputs.dataset]

        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            source = 'inm7-storage'
        else:
            source = None

        # install datasets
        dl.install(
            path=self._results['dataset_dir'], source=dataset_url[self.inputs.dataset],
            on_failure='stop')
        dl.get(
            path=dataset_dir, dataset=self._results['dataset_dir'], get_data=False, source=source,
            on_failure='stop')
        subject_dir = Path(dataset_dir, self.inputs.subject)
        dl.get(
            path=subject_dir, dataset=dataset_dir, get_data=False, source=source,
            on_failure='stop')

        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            d_dir = Path(subject_dir, 'unprocessed', 'Diffusion')
            dl.get(
                path=d_dir.parent, dataset=dataset_dir, get_data=False, source=source,
                on_failure='stop')
            anat_dir = Path(subject_dir, 'T1w')
            fs_dir = Path(subject_dir, 'T1w', self.inputs.subject)
            dl.get(
                path=anat_dir, dataset=dataset_dir, get_data=False, source=source,
                on_failure='stop')
            mni_dir = Path(subject_dir, 'MNINonLinear')
            dl.get(
                path=mni_dir, dataset=dataset_dir, get_data=False, source=source,
                on_failure='stop')

            d_files = {}
            for dirs in [98, 99]:
                for phase in ['AP', 'PA']:
                    for ftype in ['.nii.gz', '.bval', '.bvec']:
                        key = f'dir{dirs}_{phase}{ftype}'
                        d_files[key] = Path(d_dir, f'{self.inputs.subject}_dMRI_{key}')

            self._results['d_files'] = d_files.copy()
            for key, val in d_files.items():
                if val.is_symlink():
                    dl.get(path=val, dataset=d_dir.parent, source=source, on_failure='stop')
                else:
                    self._results['d_files'][key] = ''

            fs_files = {
                'lh_aparc': Path(fs_dir, 'label', 'lh.aparc.annot'),
                'rh_aparc': Path(fs_dir, 'label', 'rh.aparc.annot'),
                'lh_pial': Path(fs_dir, 'surf', 'lh.pial'),
                'rh_pial': Path(fs_dir, 'surf', 'rh.pial'),
                'lh_white': Path(fs_dir, 'surf', 'lh.white'),
                'rh_white': Path(fs_dir, 'surf', 'rh.white'),
                'lh_white_deformed': Path(fs_dir, 'surf', 'lh.white.deformed'),
                'rh_white_deformed': Path(fs_dir, 'surf', 'rh.white.deformed'),
                'lh_reg': Path(fs_dir, 'surf', 'lh.sphere.reg'),
                'rh_reg': Path(fs_dir, 'surf', 'rh.sphere.reg'),
                'lh_ribbon': Path(fs_dir, 'mri', 'lh.ribbon.mgz'),
                'rh_ribbon': Path(fs_dir, 'mri', 'rh.ribbon.mgz'),
                'ribbon': Path(fs_dir, 'mri', 'ribbon.mgz'),
                'orig': Path(fs_dir, 'mri', 'orig.mgz'),
                'eye': Path(fs_dir, 'mri', 'transforms', 'eye.dat'),
                'lh_thickness': Path(fs_dir, 'surf', 'lh.thickness'),
                'rh_thickness': Path(fs_dir, 'surf', 'rh.thickness')}
            self._results['fs_dir'] = fs_dir

            self._results['fs_files'] = fs_files.copy()
            for key, val in fs_files.items():
                if val.is_symlink():
                    dl.get(path=val, dataset=anat_dir, source=source, on_failure='stop')
                else:
                    self._results['fs_files'][key] = ''

            t1_files = {
                't1': Path(anat_dir, 'T1w_acpc_dc.nii.gz'),
                't1_restore': Path(anat_dir, 'T1w_acpc_dc_restore.nii.gz'),
                't1_restore_brain': Path(anat_dir, 'T1w_acpc_dc_restore_brain.nii.gz'),
                'bias': Path(anat_dir, 'BiasField_acpc_dc.nii.gz'),
                'fs_mask': Path(anat_dir, 'brainmask_fs.nii.gz'),
                't1_to_mni': Path(mni_dir, 'xfms', 'acpc_dc2standard.nii.gz'),
                'aseg': Path(mni_dir, 'mri', 'aseg.mgz'),
                'aparc_aseg': Path(mni_dir, 'mri', 'aparc+aseg.mgz'),
                'brainmask': Path(mni_dir, 'mri', 'brainmask.mgz'),
                'talairach_xfm': Path(mni_dir, 'mri', 'xfms', 'talairach.xfm'),
                'norm': Path(mni_dir, 'mri', 'norm.mgz')}

            self._results['t1_files'] = t1_files.copy()
            for key, val in t1_files.items():
                if val.is_symlink():
                    if key == 't1_to_mni' or key == 'aseg':
                        dl.get(path=val, dataset=mni_dir, source=source, on_failure='stop')
                    else:
                        dl.get(path=val, dataset=anat_dir, source=source, on_failure='stop')
                else:
                    self._results['t1_files'][key] = ''

        return runtime


class _SaveFeaturesInputSpec(BaseInterfaceInputSpec):
    rsfc = traits.Dict(mandatory=True, dtype=float, desc='resting-state functional connectivity')
    dfc = traits.Dict(dmandatory=True, dtype=float, desc='dynamic functional connectivity')
    rs_stats = traits.Dict(mandatory=True, dtype=float, desc='dynamic functional connectivity')
    tfc = traits.Dict({}, dtype=dict, desc='task-based functional connectivity')
    myelin = traits.Dict(mandatory=True, dtype=float, desc='myelin content estimates')
    morph = traits.Dict(mandatory=True, dtype=float, desc='morphometry features')
    output_dir = traits.Directory(mandatory=True, desc='absolute path to output directory')
    dataset = traits.Str(
        mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    dataset_dir = traits.Directory(mandatory=True, desc='absolute path to installed root dataset')
    subject = traits.Str(mandatory=True, desc='subject ID')
    overwrite = traits.Bool(mandatory=True, desc='whether to overwrite existing results')


class SaveFeatures(SimpleInterface):
    """Save extracted features"""
    input_spec = _SaveFeaturesInputSpec

    def _run_interface(self, runtime):
        Path(self.inputs.output_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(
            self.inputs.output_dir, f'{self.inputs.dataset}_{self.inputs.subject}.h5')

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
                ds_tfc = f'/tfc/level{level+1}'
                data_tfc = self.inputs.tfc[f'level{level+1}']
                write_h5(output_file, ds_tfc, data_tfc, self.inputs.overwrite)

            for stats in ['GMV', 'CS', 'CT']:
                ds_morph = f'/morphometry/{stats}/level{level+1}'
                data_morph = self.inputs.morph[f'level{level+1}_{stats}']
                write_h5(output_file, ds_morph, data_morph, self.inputs.overwrite)

            ds_myelin = f'/myelin/level{level+1}'
            data_myelin = self.inputs.myelin[f'level{level+1}']
            write_h5(output_file, ds_myelin, data_myelin, self.inputs.overwrite)

        dl.remove(dataset=self.inputs.dataset_dir, reckless='kill', on_failure='continue')

        return runtime


class _SaveDFeaturesInputSpec(BaseInterfaceInputSpec):
    count_files = traits.List(mandatory=True, desc='SC based on streamline count')
    length_files = traits.List(mandatory=True, desc='SC based on streamline length')
    output_dir = traits.Directory(mandatory=True, desc='absolute path to output directory')
    dataset = traits.Str(
        mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    dataset_dir = traits.Directory(mandatory=True, desc='absolute path to installed root dataset')
    subject = traits.Str(mandatory=True, desc='subject ID')
    overwrite = traits.Bool(mandatory=True, desc='whether to overwrite existing results')


class SaveDFeatures(SimpleInterface):
    """Save extracted diffusion features"""
    input_spec = _SaveDFeaturesInputSpec

    def _run_interface(self, runtime):
        Path(self.inputs.output_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(
            self.inputs.output_dir, f'{self.inputs.dataset}_{self.inputs.subject}.h5')

        for count_file in self.inputs.count_files:
            sc_count = pd.read_csv(count_file, header=None)
            ds_count = f'/sc/count/level{str(sc_count.shape[0])[0]}'
            write_h5(output_file, ds_count, np.array(sc_count), self.inputs.overwrite)

        for length_file in self.inputs.length_files:
            sc_length = pd.read_csv(length_file, header=None)
            ds_length = f'/sc/length/level{str(sc_length.shape[0])[0]}'
            write_h5(output_file, ds_length, np.array(sc_length), self.inputs.overwrite)

        dl.remove(dataset=self.inputs.dataset_dir, reckless='kill', on_failure='continue')

        return runtime


class _InitFeaturesInputSpec(BaseInterfaceInputSpec):
    features_dir = traits.Dict(
        mandatory=True, dtype=str, desc='absolute path to extracted features for each dataset')
    phenotypes_dir = traits.Dict(
        mandatory=True, desc='absolute path to phenotype files for each dataset')
    phenotype = traits.Str(mandatory=True, desc='phenotype to use as prediction target')


class _InitFeaturesOutputSpec(TraitedSpec):
    sublists = traits.Dict(dtype=list, desc='list of subjects available in each dataset')
    confounds = traits.Dict(dtype=dict, desc='confound values from subjects in sublists')
    phenotypes = traits.Dict(dtype=float, desc='phenotype values from subjects in sublists')
    phenotypes_perm = traits.Dict(dtype=float, desc='shuffled phenotype values for permutation')


class InitFeatures(SimpleInterface):
    """Extract available phenotype data and required list of subjects"""
    input_spec = _InitFeaturesInputSpec
    output_spec = _InitFeaturesOutputSpec

    def _run_interface(self, runtime):
        self._results['sublists'] = dict.fromkeys(self.inputs.features_dir)
        self._results['confounds'] = {}

        for dataset in self.inputs.features_dir:
            features_files = list(Path(self.inputs.features_dir[dataset]).iterdir())
            if dataset == 'HCP-A' or 'HCP-D':
                sublist = [str(file)[-19:-9] for file in features_files]
            else:
                sublist = [str(file)[-9:-3] for file in features_files]
            sublist, pheno, pheno_perm = pheno_hcp(
                dataset, self.inputs.phenotypes_dir[dataset], self.inputs.phenotype, sublist)
            sublist, self._results['confounds'] = pheno_conf_hcp(
                dataset, self.inputs.phenotypes_dir[dataset], self.inputs.features_dir[dataset],
                sublist)
            self._results['phenotypes'] = pheno
            self._results['phenotypes_perm'] = pheno_perm
            self._results['sublists'][dataset] = sublist

        return runtime


class _RegionwiseSaveInputSpec(BaseInterfaceInputSpec):
    results = traits.List(dtype=dict, desc='accuracy results')
    selected_features = traits.List(dtype=dict, desc='whether each feature is selected or not')
    output_dir = traits.Directory(mandatory=True, desc='absolute path to output directory')
    overwrite = traits.Bool(mandatory=True, desc='whether to overwrite existing results')


class RegionwiseSave(SimpleInterface):
    """Save region-wise prediction results"""
    input_spec = _RegionwiseSaveInputSpec

    def _run_interface(self, runtime):
        Path(self.inputs.output_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(self.inputs.output_dir, 'regionwise_results.h5')

        results = {key: item for d in self.inputs.results for key, item in d.items()}
        features = {key: item for d in self.inputs.selected_features for key, item in d.items()}

        for dict_data in results, features:
            for key, val in dict_data.items():
                write_h5(output_file, f'/{key}', np.array(val), self.inputs.overwrite)

        return runtime
