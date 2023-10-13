import subprocess
from os import environ
from pathlib import Path
import sys
from typing import Union

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import nipype.pipeline as pe
from nipype.interfaces import utility as niu
import numpy as np
import nibabel as nib
import pandas as pd
from nipype.interfaces import fsl, freesurfer
import bct
import datalad.api as dl

from mpp.utilities.features import (
    parcellate, fc, diffusion_mapping, score, add_subdir, atlas_files, add_annot)
from mpp.utilities.data import read_h5
from mpp.utilities.preproc import t1_files_type, fs_files_aparc, combine_4strings
from mpp.exceptions import DatasetError

base_dir = Path(__file__).resolve().parent.parent


class _RSFCInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(
        mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    rs_runs = traits.List(mandatory=True, desc='resting-state run names')
    rs_files = traits.Dict(mandatory=True, dtype=Path, desc='filenames of resting-state data')
    hcpd_b_runs = traits.Int(0, desc='number of b runs added for HCP-D subject')


class _RSFCOutputSpec(TraitedSpec):
    rsfc = traits.Dict(dtype=float, desc='resting-state functional connectivity')
    dfc = traits.Dict(dtype=float, desc='dynamic functional connectivity')
    efc = traits.Dict(dtype=float, desc='effective functional connectivity')


class RSFC(SimpleInterface):
    """Compute resting-state static and dynamic functional connectivity"""
    input_spec = _RSFCInputSpec
    output_spec = _RSFCOutputSpec

    _tr_dataset = {'HCP-YA': 0.72, 'HCP-A': 0.8, 'HCP-D': 0.8, 'ABCD': 0.8, 'UKB': 0.735}

    def _run_interface(self, runtime):
        n_runs = len(self.inputs.rs_runs) + self.inputs.hcpd_b_runs
        tavg_dict = {}
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
                tavg = parcellate(t_surf, t_vol, self.inputs.dataset, self.inputs.rs_files)
                for key, val in tavg.items():
                    if key in tavg_dict.keys():
                        tavg_dict[key] = np.hstack((tavg_dict[key], val))
                    else:
                        tavg_dict[key] = val

        self._results['rsfc'], self._results['dfc'], self._results['efc'] = fc(
            tavg_dict, t_rep=self._tr_dataset[self.inputs.dataset])

        return runtime


class _NetworkStatsInputSpec(BaseInterfaceInputSpec):
    conn = traits.Dict({}, dtype=float, desc='connectivity matrix')
    conn_files = traits.List('', desc='connectivity matrix file')


class _NetworkStatsOutputSpec(TraitedSpec):
    stats = traits.Dict(dtype=float, desc='network statistics features')


class NetworkStats(SimpleInterface):
    """Compute graph theory based network statistics using (static) resting-state functional
    connectivity or structural connectivity"""
    input_spec = _NetworkStatsInputSpec
    output_spec = _NetworkStatsOutputSpec

    def _get_conn(self, level):
        if self.inputs.conn:
            return self.inputs.conn[f'level{level}']
        else:
            conn_file = f'{str(self.inputs.conn_files[0])[:-5]}{int(level)-1}.csv'
            conn = np.array(pd.read_csv(conn_file, header=None))

        return conn

    def _run_interface(self, runtime):
        self._results['stats'] = {}
        for level in ['1', '2', '3', '4']:
            conn = self._get_conn(level)

            stre = bct.strengths_und(conn)
            betw = bct.betweenness_wei(conn)
            part = bct.participation_coef(conn, bct.community_louvain(conn, B='negative_sym')[0])
            effi = bct.efficiency_wei(conn, local=True)

            self._results['stats'][f'level{level}_strength'] = stre
            self._results['stats'][f'level{level}_betweenness'] = betw
            self._results['stats'][f'level{level}_participation'] = part
            self._results['stats'][f'level{level}_efficiency'] = effi

        return runtime


class _TFCInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(
        mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    t_runs = traits.List(mandatory=True, desc='task run names')
    t_files = traits.Dict(mandatory=True, dtype=Path, desc='filenames of task fMRI data')


class _TFCOutputSpec(TraitedSpec):
    tfc = traits.Dict(dtype=dict, desc='task-based functional connectivity')


class TFC(SimpleInterface):
    """Compute task-based functional connectivity"""
    input_spec = _TFCInputSpec
    output_spec = _TFCOutputSpec

    _task_runs = {
        'HCP-YA': ['tfMRI_EMOTION', 'tfMRI_GAMBLING', 'tfMRI_LANGUAGE', 'tfMRI_MOTOR',
                   'tfMRI_RELATIONAL', 'tfMRI_SOCIAL', 'tfMRI_WM'],
        'HCP-A': ['tfMRI_CARIT_PA', 'tfMRI_FACENAME_PA', 'tfMRI_VISMOTOR_PA'],
        'HCP-D': ['tfMRI_CARIT', 'tfMRI_EMOTION', 'tfMRI_GUESSING']}

    def _parcellate(self, run: str) -> dict:
        t_surf = nib.load(self.inputs.t_files[f'{run}_surf']).get_fdata()
        t_vol = nib.load(self.inputs.t_files[f'{run}_vol']).get_fdata()
        tavg = parcellate(t_surf, t_vol, self.inputs.dataset, self.inputs.rs_files)

        return tavg

    @staticmethod
    def _concat(tavg1: dict, tavg2: dict) -> dict:
        tavg = {}
        for key in tavg1.keys():
            tavg[key] = np.hstack((tavg1[key], tavg2[key]))

        return tavg

    def _run_interface(self, runtime):
        self._results['tfc'] = {}
        for run in self._task_runs[self.inputs.dataset]:
            if self.inputs.dataset == 'HCP-YA':
                tavg = self._concat(self._parcellate(f'{run}_LR'), self._parcellate(f'{run}_Rl'))
            elif self.inputs.dataset == 'HCP-D':
                if run == 'tfMRI_EMOTION':
                    tavg = self._parcellate(f'{run}_PA')
                else:
                    tavg = self._concat(
                        self._parcellate(f'{run}_PA'), self._parcellate(f'{run}_AP'))
            elif self.inputs.dataset == 'HCP-A':
                tavg = self._parcellate(run)
            else:
                raise DatasetError()

            self._results['tfc'][run], _, _ = fc(tavg)

        return runtime


class _MyelineEstimateInputSpec(BaseInterfaceInputSpec):
    anat_files = traits.Dict(mandatory=True, dtype=Path, desc='filenames of anatomical data')


class _MyelineEstimateOutputSpec(TraitedSpec):
    myelin = traits.Dict(dtype=float, desc='myelin content estimates')


class MyelinEstimate(SimpleInterface):
    """Extract myelin estimate from T1dividedbyT2 files"""
    input_spec = _MyelineEstimateInputSpec
    output_spec = _MyelineEstimateOutputSpec

    def _run_interface(self, runtime):
        self._results['myelin'] = {}
        myelin_l = nib.load(self.inputs.anat_files['myelin_l']).agg_data()
        myelin_r = nib.load(self.inputs.anat_files['myelin_r']).agg_data()
        myelin_surf = np.hstack((myelin_l, myelin_r))
        myelin_vol = nib.load(self.inputs.anat_files['myelin_vol']).get_fdata()

        for level in range(4):
            parc_sch_file, _, _, parc_mel_file, _ = atlas_files(Path(base_dir, 'data'), level)
            parc_sch = nib.load(parc_sch_file).get_fdata()
            parc_mel = nib.load(parc_mel_file).get_fdata()

            parc_surf = np.zeros(((level+1)*100))
            for parcel in range((level+1)*100):
                selected = myelin_surf[np.where(parc_sch == (parcel + 1))[1]]
                selected = selected[~np.isnan(selected)]
                parc_surf[parcel] = selected.mean()

            parc_mel_mask = parc_mel.nonzero()
            parc_mel = parc_mel[parc_mel.nonzero()]
            myelin_vol_masked = np.array([
                myelin_vol[parc_mel_mask[0][i], parc_mel_mask[1][i], parc_mel_mask[2][i]]
                for i in range(parc_mel_mask[0].shape[0])])
            parcels = np.unique(parc_mel).astype(int)
            parc_vol = np.zeros((parcels.shape[0]))
            for parcel in parcels:
                selected = myelin_vol_masked[np.where(parc_mel == parcel)[0]]
                selected = selected[~np.isnan(selected)]
                parc_vol[parcel-1] = selected.mean()

            self._results['myelin'][f'level{level+1}'] = np.hstack([parc_surf, parc_vol])

        return runtime


class _MorphometryInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(
        mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    anat_dir = traits.Str(mandatory=True, desc='absolute path to installed subject T1w directory')
    anat_files = traits.Dict(mandatory=True, dtype=Path, desc='filenames of anatomical data')
    subject = traits.Str(mandatory=True, desc='subject ID')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')
    simg_cmd = traits.Any(mandatory=True, desc='command for using singularity image (or not)')


class _MorphometryOutputSpec(TraitedSpec):
    morph = traits.Dict(dtype=float, desc='morphometry features')


class Morphometry(SimpleInterface):
    """Extract morphometry features from structural data"""
    input_spec = _MorphometryInputSpec
    output_spec = _MorphometryOutputSpec

    def _run_interface(self, runtime):
        tmp_dir = Path(self.inputs.work_dir, 'morph_tmp')
        tmp_dir.mkdir(parents=True, exist_ok=True)

        self._results['morph'] = {}
        for level in range(4):
            stats_surf = None
            for hemi in ['lh', 'rh']:
                annot_fs = Path(
                    base_dir, 'data', 'label',
                    f'{hemi}.Schaefer2018_{level+1}00Parcels_17Networks_order.annot')
                dl.unlock(annot_fs, dataset=base_dir.parent, on_failure='stop')
                annot_sub = Path(
                    tmp_dir, f'{hemi}.Schaefer{level+1}00Parcels_{self.inputs.subject}.annot')

                add_subdir(
                    str(tmp_dir), self.inputs.subject,
                    str(Path(self.inputs.anat_dir, self.inputs.subject)))
                annot_fs2sub = freesurfer.SurfaceTransform(
                    command=self.inputs.simg_cmd.run_cmd(
                        'mri_surf2surf', options=f'--env SUBJECTS_DIR={tmp_dir}'),
                    source_annot_file=annot_fs, out_file=annot_sub,
                    hemi=hemi, source_subject='fsaverage',
                    target_subject=self.inputs.subject, subjects_dir=tmp_dir)
                annot_fs2sub.run()

                hemi_table = Path(tmp_dir, f'{hemi}.fs_stats')
                subprocess.run(
                    self.inputs.simg_cmd.run_cmd(
                        'mris_anatomical_stats',
                        options=f'--env SUBJECTS_DIR={self.inputs.anat_dir}').split() + ['-a',
                    str(annot_sub), '-noglobal', '-f', str(hemi_table), self.inputs.subject, hemi],
                    env=dict(environ, **{'SUBJECTS_DIR': self.inputs.anat_dir}), check=True)

                hemi_stats = pd.read_table(
                    hemi_table, header=0, skiprows=np.arange(51), delim_whitespace=True)
                hemi_stats.drop([0], inplace=True)  # exclude medial wall
                if stats_surf is None:
                    stats_surf = hemi_stats
                else:
                    stats_surf = pd.concat([stats_surf, hemi_stats])
            self._results['morph'][f'level{level+1}_CS'] = stats_surf['SurfArea'].values
            self._results['morph'][f'level{level+1}_CT'] = stats_surf['ThickAvg'].values

            seg_file = Path(base_dir, 'data', 'atlas', f'Tian_Subcortex_S{level+1}_3T.nii.gz')
            seg_up_file = Path(tmp_dir, f'S{level}_upsampled.nii.gz')
            resos = {'HCP-YA': 0.7, 'HCP-A': 0.8, 'HCP-D': 0.8}
            flt = fsl.FLIRT(
                command=self.inputs.simg_cmd.run_cmd(cmd='flirt'),
                in_file=seg_file, reference=self.inputs.anat_files['t1_vol'], out_file=seg_up_file,
                apply_isoxfm=resos[self.inputs.dataset], interp='nearestneighbour')
            flt.run()

            sub_table = 'subcortex.stats'
            ss = freesurfer.SegStats(
                command=self.inputs.simg_cmd.run_cmd(
                    'mri_segstats', options=f'--env SUBJECTS_DIR={self.inputs.anat_dir}'),
                segmentation_file=seg_up_file, in_file=self.inputs.anat_files['t1_vol'],
                summary_file=sub_table, subjects_dir=self.inputs.anat_dir)
            ss.run()

            stats_vol = pd.read_table(
                sub_table, header=0, skiprows=np.arange(50), delim_whitespace=True)
            stats_vol.drop([0], inplace=True)

            self._results['morph'][f'level{level+1}_GMV'] = np.concatenate((
                stats_surf['GrayVol'].values, stats_vol['Volume_mm3'].values))

        return runtime


class _CVFeaturesInputSpec(BaseInterfaceInputSpec):
    features_dir = traits.Dict(
        mandatory=True, dtype=str, desc='absolute path to extracted features for each dataset')
    sublists = traits.Dict(
        mandatory=True, dtype=list, desc='list of subjects available in each dataset')
    cv_split = traits.Dict(
        mandatory=True, dtype=list, desc='list of subjects in the test split of each fold')
    # cv_split_perm = traits.Dict(mandatory=True, dtype=list, desc='list of permuted subjects')
    repeat = traits.Int(mandatory=True, desc='current repeat of cross-validation')
    fold = traits.Int(mandatory=True, desc='current fold in the repeat')
    level = traits.Str(mandatory=True, desc='parcellation level')
    config = traits.Dict({}, desc='configuration settings')


class _CVFeaturesOutputSpec(TraitedSpec):
    embeddings = traits.Dict(desc='embeddings for gradients')
    params = traits.Dict(desc='parameters for anatomical connectivity')
    repeat = traits.Int(desc='current repeat of cross-validation')
    fold = traits.Int(desc='current fold in the repeat')
    level = traits.Str(desc='parcellation level')


class CVFeatures(SimpleInterface):
    """Compute gradient embedding (diffusion mapping) and
    anatomical connectivity (structural co-registration)"""
    input_spec = _CVFeaturesInputSpec
    output_spec = _CVFeaturesOutputSpec

    def _extract_features(self) -> dict:
        all_sub = sum(self.inputs.sublists.values(), [])
        image_features = dict.fromkeys(all_sub)

        for subject in all_sub:
            dataset = [
                key for key in self.inputs.sublists if subject in self.inputs.sublists[key]][0]
            if dataset in ['HCP-A', 'HCP-D']:
                feature_file = Path(
                    self.inputs.features_dir[dataset], f'{dataset}_{subject}_V1_MR.h5')
            else:
                feature_file = Path(self.inputs.features_dir[dataset], f'{dataset}_{subject}.h5')

            feature_sub = {
                'rsfc': read_h5(feature_file, f'/rsfc/level{self.inputs.level}'),
                'myelin': read_h5(feature_file, f'/myelin/level{self.inputs.level}')}
            for stats in ['GMV', 'CS', 'CT']:
                ds_morph = f'/morphometry/{stats}/level{self.inputs.level}'
                feature_sub[stats] = read_h5(feature_file, ds_morph)
            image_features[subject] = feature_sub

        return image_features

    def _compute_features(
            self, image_features: dict, cv_split: dict,
            repeat: int) -> tuple[np.ndarray, pd.DataFrame]:
        all_sub = sum(self.inputs.sublists.values(), [])
        n_folds = int(self.inputs.config['n_folds'])

        test_sub = cv_split[f'repeat{repeat}_fold{self.inputs.fold}']
        val_sub = cv_split[f'repeat{repeat}_fold{(self.inputs.fold+1)%n_folds}']
        testval_sub = np.concatenate((val_sub, test_sub))
        train_sub = [subject for subject in all_sub if subject not in testval_sub]

        embed = diffusion_mapping(image_features, train_sub, 'rsfc')
        for feature in ['GMV', 'CS', 'CT', 'myelin']:
            params = score(image_features, train_sub, feature)

        return embed, params

    def _run_interface(self, runtime):
        image_features = self._extract_features()
        self._results['embeddings'] = {}
        self._results['params'] = {}

        embed, params = self._compute_features(
            image_features, self.inputs.cv_split, self.inputs.repeat)
        self._results['embeddings']['embedding'] = embed
        self._results['params']['params'] = params

        # assuming n_repeats_perm = n_repeats x 10
        # n_perm_check = int(self.inputs.config['n_repeats']) * 10
        # if int(self.inputs.config['n_repeats_perm']) == n_perm_check:
        #    for repeat_split in range(10):
        #        repeat = int(self.inputs.repeat) * 10 + repeat_split
        #        embed, params = self._compute_features(
        #            image_features, self.inputs.cv_split_perm, repeat)
        #        self._results['embeddings'][f'repeat{repeat}'] = embed
        #        self._results['params'][f'repeat{repeat}'] = params

        self._results['repeat'] = self.inputs.repeat
        self._results['fold'] = self.inputs.fold
        self._results['level'] = self.inputs.level

        return runtime


class _CombineAtlasInputSpec(BaseInterfaceInputSpec):
    cort_file = traits.File(mandatory=True, exists=True, desc='cortex atlas in T1 space')
    subcort_file = traits.File(mandatory=True, exists=True, desc='subcortex atlas in t1 space')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')
    level = traits.Int(mandatory=True, desc='parcellation level (0 to 3)')


class _CombineAtlasOutputSpec(TraitedSpec):
    combined_file = traits.File(exists=True, desc='combined atlas in T1 space')


class CombineAtlas(SimpleInterface):
    """combine cortex and subcortex atlases in T1 space"""
    input_spec = _CombineAtlasInputSpec
    output_spec = _CombineAtlasOutputSpec

    def _run_interface(self, runtime):
        atlas_img = nib.load(self.inputs.cort_file)
        atlas_cort = atlas_img.get_fdata()
        atlas_subcort = nib.load(self.inputs.subcort_file).get_fdata()
        atlas = np.zeros(atlas_cort.shape)

        cort_parcels = np.unique(atlas_cort[np.where(atlas_cort > 1000)])
        for parcel in cort_parcels:
            if 1000 < parcel < 2000:  # lh
                atlas[atlas_cort == parcel] = parcel - 1000
            elif parcel > 2000:  # rh
                atlas[atlas_cort == parcel] = parcel - 2000 + (len(cort_parcels) - 1) / 2

        for parcel in np.unique(atlas_subcort):
            if parcel != 0:
                atlas[atlas_subcort == parcel] = parcel + 100 * (self.inputs.level + 1)

        self._results['combined_file'] = Path(
            self.inputs.work_dir, f'atlas_combine_level{self.inputs.level}.nii.gz')
        nib.save(
            nib.Nifti1Image(atlas, header=atlas_img.header, affine=atlas_img.affine),
            self._results['combined_file'])

        return runtime


class _SCInputSpec(BaseInterfaceInputSpec):
    atlas_file = traits.File(mandatory=True, exists=True, desc='combined atlas in T1 space')
    tck_file = traits.File(mandatory=True, exists=True, desc='tracks file')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')
    level = traits.Int(mandatory=True, desc='parcellation level (0 to 3)')
    simg_cmd = traits.Any(mandatory=True, desc='command for using singularity image (or not)')


class _SCOutputSpec(TraitedSpec):
    count_file = traits.File(exists=True, desc='SC based on streamline count')
    length_file = traits.File(exists=True, desc='SC based on streamline length')


class SC(SimpleInterface):
    """Compute structural connectivity based on streamline count and length-scaled count"""
    input_spec = _SCInputSpec
    output_spec = _SCOutputSpec

    def _run_interface(self, runtime):
        self._results['count_file'] = Path(
            self.inputs.work_dir, f'sc_count_level{self.inputs.level}.csv')
        sc_count = self.inputs.simg_cmd.run_cmd('tck2connectome').split() + [
            '-assignment_radial_search', '2', '-symmetric', '-nthreads', '0',
            str(self.inputs.tck_file), str(self.inputs.atlas_file),
            str(self._results['count_file'])]
        if not self._results['count_file'].is_file():
            subprocess.run(sc_count, check=True)

        self._results['length_file'] = Path(
            self.inputs.work_dir, f'sc_length_level{self.inputs.level}.csv')
        sc_length = self.inputs.simg_cmd.run_cmd('tck2connectome').split() + [
            '-assignment_radial_search', '2', '-scale_length',
            '-stat_edge', 'mean', '-symmetric', '-nthreads', '0',
            str(self.inputs.tck_file), str(self.inputs.atlas_file),
            str(self._results['length_file'])]
        if not self._results['length_file'].is_file():
            subprocess.run(sc_length, check=True)

        return runtime


class _SCWFInputSpec(BaseInterfaceInputSpec):
    subject = traits.Str(mandatory=True, desc='subject ID')
    dataset = traits.Str(
        mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')
    simg_cmd = traits.Any(mandatory=True, desc='command for using singularity image (or not)')


class _SCWFOutputSpec(TraitedSpec):
    sc_wf = traits.Any(desc='structural connectivity workflow')


class SCWF(SimpleInterface):
    """Generate a workflow for computing structural connectivity"""
    input_spec = _SCWFInputSpec
    output_spec = _SCWFOutputSpec

    def _run_interface(self, runtime):
        tmp_dir = Path(self.inputs.work_dir, 'sc_tmp')
        tmp_dir.mkdir(parents=True, exist_ok=True)

        self._results['sc_wf'] = pe.Workflow('sc_wf', base_dir=self.inputs.work_dir)
        inputnode = pe.Node(
            niu.IdentityInterface(fields=['tck_file', 't1_files', 'fs_files', 'fs_dir']),
            name='inputnode')
        sub_dir = pe.Node(
            niu.Function(function=add_subdir, output_names=['sub_dir']), name='sub_dir')
        sub_dir.inputs.sub_dir = str(tmp_dir)
        sub_dir.inputs.subject = self.inputs.subject
        split_t1_files = pe.Node(
            niu.Function(
                function=t1_files_type, output_names=[
                    't1', 't1_restore', 't1_restore_brain', 'bias', 'fs_mask', 'xfm']),
            name='split_t1_files')
        split_fs_files = pe.Node(
            niu.Function(
                function=fs_files_aparc, output_names=[
                    'lh_aparc', 'rh_aparc', 'lh_white', 'rh_white', 'lh_pial', 'rh_pial',
                    'lh_ribbon', 'rh_ribbon', 'ribbon']),
            name='split_fs_files')
        split_atlas = pe.Node(
            niu.Function(
                function=atlas_files, output_names=[
                    'cort_atlas', 'lh_cort_annot', 'rh_cort_annot', 'subcort_atlas', 'level']),
            iterables=[('level', [0, 1, 2, 3])], name='split_atlas')
        split_atlas.inputs.data_dir = str(Path(base_dir, 'data'))

        std2t1 = pe.Node(
            fsl.InvWarp(command=self.inputs.simg_cmd.run_cmd('invwarp')), name='std2t1')
        subcort_t1 = pe.Node(
            fsl.ApplyWarp(
                command=self.inputs.simg_cmd.run_cmd('applywarp'), interp='nn', relwarp=True),
            name='subcort_t1')
        lh_cort_sub = pe.Node(
            freesurfer.SurfaceTransform(
                command=self.inputs.simg_cmd.run_cmd(
                    'mri_surf2surf', options=f'--env SUBJECTS_DIR={tmp_dir}'), hemi='lh',
                source_subject='fsaverage', target_subject=self.inputs.subject),
            name='lh_cort_sub')
        rh_cort_sub = pe.Node(
            freesurfer.SurfaceTransform(
                command=self.inputs.simg_cmd.run_cmd(
                    'mri_surf2surf', options=f'--env SUBJECTS_DIR={tmp_dir}'), hemi='rh',
                source_subject='fsaverage', target_subject=self.inputs.subject),
            name='rh_cort_sub')
        copy_annot = pe.Node(
            niu.Function(function=add_annot, output_names=['annot_args']), name='copy_annot')
        copy_annot.inputs.subject = self.inputs.subject
        aseg_out = pe.Node(
            niu.Function(function=combine_4strings, output_names=['str_out']), name='aseg_out')
        aseg_out.inputs.str1 = str(tmp_dir)
        aseg_out.inputs.str2 = '/aparc2aseg_'
        aseg_out.inputs.str4 = '.nii.gz'
        cort_aseg = pe.Node(
            freesurfer.Aparc2Aseg(
                command=self.inputs.simg_cmd.run_cmd(
                    'mri_aparc2aseg', options=f'--env SUBJECTS_DIR={tmp_dir}'),
                subject_id=self.inputs.subject), name='cort_aseg')
        cort_t1 = pe.Node(
            fsl.FLIRT(command=self.inputs.simg_cmd.run_cmd('flirt'), interp='nearestneighbour'),
            name='cort_t1')
        combine_atlas = pe.Node(CombineAtlas(work_dir=tmp_dir), name='combine_atlas')
        resos = {'HCP-YA': 1.25, 'HCP-A': 1.5, 'HCP-D': 1.5}
        atlas_ds = pe.Node(fsl.FLIRT(
            command=self.inputs.simg_cmd.run_cmd(cmd='flirt'), interp='nearestneighbour',
            apply_isoxfm=resos[self.inputs.dataset], datatype='int'), name='atlas_ds')

        sc = pe.Node(SC(work_dir=tmp_dir, simg_cmd=self.inputs.simg_cmd), name='sc')
        outputnode = pe.JoinNode(
            niu.IdentityInterface(fields=['count_files', 'length_files', 'atlas_files']),
            name='outputnode', joinfield=['count_files', 'length_files', 'atlas_files'],
            joinsource='split_atlas')

        self._results['sc_wf'].connect([
            (inputnode, sub_dir, [('fs_dir', 'fs_dir')]),
            (inputnode, split_t1_files, [('t1_files', 't1_files')]),
            (inputnode, split_fs_files, [('fs_files', 'fs_files')]),
            (split_t1_files, std2t1, [('xfm', 'warp'), ('t1_restore_brain', 'reference')]),
            (split_t1_files, subcort_t1, [('t1_restore_brain', 'ref_file')]),
            (std2t1, subcort_t1, [('inverse_warp', 'field_file')]),
            (split_atlas, subcort_t1, [('subcort_atlas', 'in_file')]),
            (sub_dir, lh_cort_sub, [('sub_dir', 'subjects_dir')]),
            (split_atlas, lh_cort_sub, [('lh_cort_annot', 'source_annot_file')]),
            (sub_dir, rh_cort_sub, [('sub_dir', 'subjects_dir')]),
            (split_atlas, rh_cort_sub, [('rh_cort_annot', 'source_annot_file')]),
            (sub_dir, copy_annot, [('sub_dir', 'sub_dir')]),
            (lh_cort_sub, copy_annot, [('out_file', 'lh_annot')]),
            (rh_cort_sub, copy_annot, [('out_file', 'rh_annot')]),
            (split_atlas, aseg_out, [('level', 'str3')]),
            (sub_dir, cort_aseg, [('sub_dir', 'subjects_dir')]),
            (split_fs_files, cort_aseg, [
                ('lh_aparc', 'lh_annotation'), ('rh_aparc', 'rh_annotation'),
                ('lh_pial', 'lh_pial'), ('rh_pial', 'rh_pial'), ('lh_ribbon', 'lh_ribbon'),
                ('rh_ribbon', 'rh_ribbon'), ('lh_white', 'lh_white'), ('rh_white', 'rh_white'),
                ('ribbon', 'ribbon')]),
            (copy_annot, cort_aseg, [('annot_args', 'args')]),
            (aseg_out, cort_aseg, [('str_out', 'out_file')]),
            (split_t1_files, cort_t1, [('t1_restore_brain', 'reference')]),
            (cort_aseg, cort_t1, [('out_file', 'in_file')]),
            (split_atlas, combine_atlas, [('level', 'level')]),
            (subcort_t1, combine_atlas, [('out_file', 'subcort_file')]),
            (cort_t1, combine_atlas, [('out_file', 'cort_file')]),
            (inputnode, sc, [('tck_file', 'tck_file')]),
            (split_atlas, sc, [('level', 'level')]),
            (combine_atlas, atlas_ds, [
                ('combined_file', 'in_file'), ('combined_file', 'reference')]),
            (atlas_ds, sc, [('out_file', 'atlas_file')]),
            (atlas_ds, outputnode, [('out_file', 'atlas_files')]),
            (sc, outputnode, [('count_file', 'count_files'), ('length_file', 'length_files')])])

        return runtime


class _FAMDInputSpec(BaseInterfaceInputSpec):
    atlas_files = traits.List(mandatory=True, exists=True, desc='combined atlas in T1 space')
    fa_file = traits.File(mandatory=True, exists=True, desc='FA file')
    md_file = traits.File(mandatory=True, exists=True, desc='MD file')


class _FAMDOutputSpec(TraitedSpec):
    fa = traits.Dict(desc='region-wise FA values')
    md = traits.Dict(desc='region-wise MD values')


class FAMD(SimpleInterface):
    """Compute region-wise FA and MD"""
    input_spec = _FAMDInputSpec
    output_spec = _FAMDOutputSpec

    @staticmethod
    def _extract_data(atlas, data):
        parcels = np.unique(atlas).astype(int)
        level = parcels.max() // 100
        parc = np.zeros(parcels.shape[0])

        for parcel in range(parcels.shape[0]):
            selected = data[np.where(atlas == parcel)[0]]
            selected = selected[~np.isnan(selected)]
            selected = selected[
                np.where(np.abs(selected.mean(axis=0)) >= sys.float_info.epsilon)[0]]
            parc[parcel-1] = selected.mean()

        return parc, level

    def _run_interface(self, runtime):
        self._results['fa'] = {}
        self._results['md'] = {}
        data_fa = nib.load(self.inputs.fa_file).get_fdata()
        data_md = nib.load(self.inputs.md_file).get_fdata()

        for atlas_file in self.inputs.atlas_files:
            atlas = nib.load(atlas_file).get_fdata()
            parc_fa, level = self._extract_data(atlas, data_fa)
            parc_md, _ = self._extract_data(atlas, data_md)

            self._results['fa'][f'level{level}'] = parc_fa
            self._results['md'][f'level{level}'] = parc_md

        return runtime
