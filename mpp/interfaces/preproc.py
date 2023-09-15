from pathlib import Path
from os import getenv
from typing import Union
import subprocess
import logging

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import nipype.pipeline as pe
from nipype.interfaces import utility as niu
import numpy as np
import datalad.api as dl
import pandas as pd
from nipype.interfaces import fsl, freesurfer, workbench
import nibabel as nib

from mpp.utilities.preproc import (
    d_files_dirsphase, d_files_type, t1_files_type, fs_files_type, update_d_files,
    flatten_list, create_2item_list, last_list_item,
    combine_2strings, combine_4strings, diff_res,
    flirt_bbr_sch, bet_nodif_mask)
from mpp.exceptions import DatasetError

logging.getLogger('datalad').setLevel(logging.WARNING)


class _ExtractB0InputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(mandatory=True,
                         desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    bval_file = traits.File(mandatory=True, desc='absolute path to the bval file')
    dwi_file = traits.File(mandatory=True, desc='absolute path to the DWI file')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')
    rescale = traits.Bool(False, desc='if b0dist should be applied on rescaled data')
    simg_cmd = traits.Any(mandatory=True, desc='command for using singularity image (or not)')


class _ExtractB0OutputSpec(TraitedSpec):
    roi_files = traits.List(dtpye=Path, desc='filenames of B0 files')
    pos_files = traits.List(dtpye=Path, desc='filenames of B0 files with positive phase encoding')
    neg_files = traits.List(dtpye=Path, desc='filenames of B0 files with negative phase encoding')


class ExtractB0(SimpleInterface):
    """Extract b0 slices"""
    input_spec = _ExtractB0InputSpec
    output_spec = _ExtractB0OutputSpec

    def _extract_b0(self, b0dist: Union[int, None] = None) -> list:
        b0maxbval = 50  # values below this will be considered as b0s
        bvals = pd.read_csv(
            self.inputs.bval_file, header=None, delim_whitespace=True).squeeze('rows')

        if b0dist is None:
            dist_count = 0
            roi_files = [self.inputs.dwi_file]
        else:
            dist_count = b0dist + 1
            roi_files = []
        dim4 = nib.load(self.inputs.dwi_file).header.get_data_shape()[3]
        vol_count = 0

        for b in bvals:
            roi_file = Path(
                self.inputs.work_dir, f'roi{vol_count}_{Path(self.inputs.dwi_file).name}')
            if b < b0maxbval and b0dist is None:
                roi = fsl.ExtractROI(
                    command=self.inputs.simg_cmd.run_cmd('fslroi'),
                    in_file=self.inputs.dwi_file, t_min=dist_count, t_size=1, roi_file=roi_file)
                roi.run()
                roi_files.append(roi_file)
            elif b < b0maxbval and vol_count < dim4 and dist_count > b0dist:
                roi = fsl.ExtractROI(
                    command=self.inputs.simg_cmd.run_cmd('fslroi'),
                    in_file=self.inputs.dwi_file, t_min=vol_count, t_size=1, roi_file=roi_file)
                roi.run()
                roi_files.append(roi_file)
                dist_count = 0
            dist_count = dist_count + 1
            vol_count = vol_count + 1

        return roi_files

    def _split_pos_neg(self, roi_files: list) -> tuple[list, ...]:
        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            pos_files = [roi_file for roi_file in roi_files if 'AP' in str(roi_file)]
            neg_files = [roi_file for roi_file in roi_files if 'PA' in str(roi_file)]
        else:
            raise DatasetError()

        return pos_files, neg_files

    def _run_interface(self, runtime):
        if not self.inputs.rescale:
            self._results['roi_files'] = self._extract_b0()
        else:
            b0dist = 45  # minimum distance between b0s
            self._results['roi_files'] = self._extract_b0(b0dist=b0dist)
            self._results['pos_files'], self._results['neg_files'] = self._split_pos_neg(
                self._results['roi_files'])

        return runtime


class _RescaleInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(
        mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    scale_files = traits.List(mandatory=True, dtype=str, desc='filenames of scale files')
    d_files = traits.Dict(mandatory=True, dtype=Path, desc='filenames of diffusion data')
    simg_cmd = traits.Any(mandatory=True, desc='command for using singularity image (or not)')


class _RescaleOutputSpec(TraitedSpec):
    rescaled_files = traits.List(dtype=str, desc='filenames of rescaled DWI images')


class Rescale(SimpleInterface):
    """Rescale DWI images, except the first one"""
    input_spec = _RescaleInputSpec
    output_spec = _RescaleOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            key = 'dir98_AP'
            keys = ['dir98_PA', 'dir99_AP', 'dir99_PA']
        else:
            raise DatasetError()

        rescale_file = [s_file for s_file in self.inputs.scale_files if key in s_file]
        rescale = pd.read_csv(rescale_file[0], header=None).squeeze()
        self._results['rescaled_files'] = [
            self.inputs.d_files[d_key] for d_key in self.inputs.d_files if key in d_key]

        for key in keys:
            scale_file = [s_file for s_file in self.inputs.scale_files if key in s_file]
            scale = pd.read_csv(scale_file[0], header=None).squeeze()
            d_file = [self.inputs.d_files[d_key] for d_key in self.inputs.d_files if key in d_key]
            maths = fsl.ImageMaths(
                command=self.inputs.simg_cmd.run_cmd('fslmaths'),
                in_file=d_file[0], args=f'-mul {rescale} -div {scale}')
            maths.run()
            self._results['rescaled_files'].append(maths.aggregate_outputs().out_file)

        return runtime


class _PrepareTopupInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(
        mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    roi_files = traits.List(mandatory=True, dtpye=Path, desc='filenames of B0 files')
    d_files = traits.Dict(mandatory=True, dtype=Path, desc='filenames of diffusion data')
    pos_b0_file = traits.File(mandatory=True, desc='merged positive b0 file')


class _PrepareTopupOutputSpec(TraitedSpec):
    enc_dir = traits.List(dtype=str, desc='encoding directions for each b0')
    ro_time = traits.Float(desc='readout time')
    indices_t = traits.List(dtype=int, desc='indices based on time dimension of b0 files')


class PrepareTopup(SimpleInterface):
    """Prepare parameters for FSL Topup"""
    input_spec = _PrepareTopupInputSpec
    output_spec = _PrepareTopupOutputSpec

    def _run_interface(self, runtime):
        # encoding direction
        self._results['enc_dir'] = []
        for roi_file in self.inputs.roi_files:
            if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
                if 'AP' in str(roi_file):
                    self._results['enc_dir'].append('y')
                elif 'PA' in str(roi_file):
                    self._results['enc_dir'].append('y-')

        # readout time
        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            # dim2 for AP/PA encoding
            dim_p = nib.load(
                self.inputs.d_files['dir98_AP.nii.gz']).header.get_data_shape()[1]
            # echo spacing from protocol files in 'HCP_VE11C_Prisma_2019.01.14' from HCP Lifespan
            echospacing = 0.69
        else:
            raise DatasetError()
        self._results['ro_time'] = round(echospacing * (dim_p - 1) / 1000, 6)

        # time dimension
        self._results['indices_t'] = [
            1, nib.load(self.inputs.pos_b0_file).header.get_data_shape()[3] + 1]

        return runtime


class _MergeBFilesInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(mandatory=True,
                         desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    bval_files = traits.List(mandatory=True, dtype=Path, desc='list of bval files to merge')
    bvec_files = traits.List(mandatory=True, dtype=Path, desc='list of bvec files to merge')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')


class _MergeBFilesOutputSpec(TraitedSpec):
    bval_merged = traits.File(exists=True, desc='merged bval file')
    bvec_merged = traits.File(exists=True, desc='merged bvec file')


class MergeBFiles(SimpleInterface):
    """Merge bval and bvec files respectively"""
    input_spec = _MergeBFilesInputSpec
    output_spec = _MergeBFilesOutputSpec

    def _run_interface(self, runtime):
        bvals = pd.DataFrame()
        bvecs = pd.DataFrame()

        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            keys = ['dir98_AP', 'dir98_PA', 'dir99_AP', 'dir99_PA']
            for key in keys:
                bval_file = [b_file for b_file in self.inputs.bval_files if key in str(b_file)]
                bvec_file = [b_file for b_file in self.inputs.bvec_files if key in str(b_file)]
                bvals = pd.concat(
                    [bvals, pd.read_csv(bval_file[0], delim_whitespace=True, header=None)], axis=1)
                bvecs = pd.concat(
                    [bvecs, pd.read_csv(bvec_file[0], delim_whitespace=True, header=None)], axis=1)
        else:
            raise DatasetError()

        self._results['bval_merged'] = Path(self.inputs.work_dir, 'merged.bval')
        self._results['bvec_merged'] = Path(self.inputs.work_dir, 'merged.bvec')
        bvals.to_csv(self._results['bval_merged'], sep='\t', header=False, index=False)
        bvecs.to_csv(self._results['bvec_merged'], sep='\t', header=False, index=False)

        return runtime


class _EddyIndexInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(
        mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    roi_files = traits.List(mandatory=True, dtype=Path, desc='filenames of B0 files')
    dwi_files = traits.List(mandatory=True, dtype=Path, desc='filenames of rescaled DWI images')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')


class _EddyIndexOutputSpec(TraitedSpec):
    index_file = traits.File(exists=True, desc='filename of index file')


class EddyIndex(SimpleInterface):
    """Create index file for eddy correction"""
    input_spec = _EddyIndexInputSpec
    output_spec = _EddyIndexOutputSpec

    def _run_interface(self, runtime):
        rois = [
            int(str(roi_file.name).lstrip('roi').split('_')[0]) for roi_file in
            self.inputs.roi_files]
        unique_rois = np.sort(np.unique(rois))

        indices = []
        pos_count = 0
        neg_count = 0
        vol_prev = 0
        for roi_file in self.inputs.roi_files:
            if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
                vol_curr = int(str(roi_file.name).lstrip('roi').split('_')[0])

                keys = ['dir98_AP', 'dir98_PA', 'dir99_AP', 'dir99_PA']
                key = [k for k in keys if k in str(roi_file)]
                dwi_file = [d_file for d_file in self.inputs.dwi_files if key[0] in str(d_file)]
                dim4 = nib.load(dwi_file[0]).header.get_data_shape()[3]

                for _ in range(vol_prev, vol_curr):
                    if 'AP' in str(roi_file):
                        indices.append(pos_count)
                    elif 'PA' in str(roi_file):
                        indices.append(neg_count)

                if 'AP' in str(roi_file):
                    pos_count = pos_count + 1
                elif 'PA' in str(roi_file):
                    neg_count = neg_count + 1

                if vol_curr == unique_rois[-1]:
                    for _ in range(vol_curr, dim4):
                        if 'AP' in str(roi_file):
                            indices.append(pos_count)
                        elif 'PA' in str(roi_file):
                            indices.append(neg_count)

                vol_prev = vol_curr

            else:
                raise DatasetError()

        self._results['index_file'] = Path(self.inputs.work_dir, 'index.txt')
        pd.DataFrame(indices).to_csv(
            self._results['index_file'], sep='\t', header=False, index=False)

        return runtime


class _EddyPostProcInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(
        mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    bval_files = traits.List(mandatory=True, dtype=Path, desc='list of bval files to merge')
    bvec_files = traits.List(mandatory=True, dtype=Path, desc='list of bvec files to merge')
    eddy_corrected_file = traits.File(
        mandatory=True, exists=True, desc='filename of eddy corrected image')
    eddy_bvecs_file = traits.File(
        mandatory=True, exists=True, desc='filename of eddy corrected bvecs')
    rescaled_files = traits.List(
        mandatory=True, dtype=Path, desc='filenames of rescaled DWI images')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')
    simg_cmd = traits.Any(mandatory=True, desc='command for using singularity image (or not)')


class _EddyPostProcOutputSpec(TraitedSpec):
    combined_dwi_file = traits.File(exists=True, desc='combined DWI data')
    rot_bvals = traits.File(exists=True, desc='average rotated bvals')
    rot_bvecs = traits.File(exists=True, desc='average rotated bvecs')


class EddyPostProc(SimpleInterface):
    """Post-eddy processing: combine output files and rotate bval/bvec from eddy correction"""
    input_spec = _EddyPostProcInputSpec
    output_spec = _EddyPostProcOutputSpec

    def _generate_files(
            self, keys: list, dirs: str) -> tuple[Path, Path, Path, Path, pd.DataFrame, list]:
        bvals = pd.DataFrame()
        bvecs = pd.DataFrame()
        corrvols = []
        tsizes = []
        for key in keys:
            bval_file = [b_file for b_file in self.inputs.bval_files if key in str(b_file)]
            bvec_file = [b_file for b_file in self.inputs.bvec_files if key in str(b_file)]
            bval = pd.read_csv(bval_file[0], delim_whitespace=True, header=None)
            bvals = pd.concat([bvals, bval], axis=1)
            bvecs = pd.concat(
                [bvecs, pd.read_csv(bvec_file[0], delim_whitespace=True, header=None)], axis=1)

            rescaled_file = [d_file for d_file in self.inputs.rescaled_files if key in str(d_file)]
            dim4 = nib.load(rescaled_file[0]).header.get_data_shape()[3]
            corrvols.append([dim4, dim4])
            tsizes.append(bval.shape[1])

        bval_merged = Path(self.inputs.work_dir, f'{dirs}.bval')
        bvals.to_csv(bval_merged, sep='\t', header=False, index=False)
        bvec_merged = Path(self.inputs.work_dir, f'{dirs}.bvec')
        bvecs.to_csv(bvec_merged, sep='\t', header=False, index=False)
        corrvols_file = Path(self.inputs.work_dir, f'{dirs}_volnum.txt')
        pd.DataFrame(corrvols).to_csv(corrvols_file, sep='\t', header=False, index=False)

        bval_tsize = bvals.shape[1]
        extract_roi = fsl.ExtractROI(
            command=self.inputs.simg_cmd.run_cmd('fslroi'), in_file=self.inputs.eddy_corrected_file,
            t_size=bval_tsize)
        if dirs == 'pos':
            extract_roi.inputs.t_min = 0
        elif dirs == 'neg':
            extract_roi.inputs.t_min = bval_tsize
        extract_roi.run()

        return (
            extract_roi.aggregate_outputs().roi_file, bval_merged, bvec_merged, corrvols_file,
            bvals, tsizes)

    def _rotate_b(
            self, pos_tsize: list, neg_tsize: list, pos_bvals: pd.DataFrame,
            neg_bvals: pd.DataFrame) -> None:
        rot_bvecs = pd.read_csv(self.inputs.eddy_bvecs_file, delim_whitespace=True, header=None)
        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            pos_rot_bvecs = np.zeros((3, sum(pos_tsize)))
            neg_rot_bvecs = np.zeros((3, sum(neg_tsize)))
            break_pos = [pos_tsize[0], pos_tsize[0] + neg_tsize[0], sum(pos_tsize) + neg_tsize[0]]
            pos_rot_bvecs[:, :pos_tsize[0]] = rot_bvecs.iloc[:, :break_pos[0]]
            neg_rot_bvecs[:, :neg_tsize[0]] = rot_bvecs.iloc[:, break_pos[0]:break_pos[1]]
            pos_rot_bvecs[:, pos_tsize[0]:] = rot_bvecs.iloc[:, break_pos[1]:break_pos[2]]
            neg_rot_bvecs[:, neg_tsize[0]:] = rot_bvecs.iloc[:, break_pos[2]:]

            avg_bvals = np.zeros((sum(pos_tsize)), dtype='i4')
            avg_bvecs = np.zeros((3, sum(pos_tsize)))
            for i in range(sum(pos_tsize)):
                pos_bvec = np.array(
                    pos_bvals.iloc[:, i]) * np.array(pos_rot_bvecs[:, i]).reshape((3, 1))
                neg_bvec = np.array(
                    neg_bvals.iloc[:, i]) * np.array(neg_rot_bvecs[:, i]).reshape((3, 1))
                bvec_sum = (np.dot(pos_bvec, pos_bvec.T) + np.dot(neg_bvec, neg_bvec.T)) / 2
                eigvals, eigvecs = np.linalg.eig(bvec_sum)
                eigvalmax = np.argmax(eigvals)
                avg_bvals[i] = np.rint(eigvals[eigvalmax] ** 0.5)
                avg_bvecs[:, i] = eigvecs[:, eigvalmax]

            self._results['rot_bvals'] = Path(self.inputs.work_dir, 'rotated.bval')
            self._results['rot_bvecs'] = Path(self.inputs.work_dir, 'rotated.bvec')
            pd.DataFrame(avg_bvals).T.to_csv(
                self._results['rot_bvals'], sep=' ', header=False, index=False)
            pd.DataFrame(avg_bvecs).to_csv(
                self._results['rot_bvecs'], sep=' ', header=False, index=False,
                float_format='%0.16f')

        else:
            raise DatasetError()

    def _run_interface(self, runtime):
        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            pos_keys = ['dir98_AP', 'dir99_AP']
            neg_keys = ['dir98_PA', 'dir99_PA']
        else:
            raise DatasetError()

        pos_dwi, pos_bval, pos_bvec, pos_corrvols, pos_bvals, pos_tsize = self._generate_files(
            pos_keys, 'pos')
        neg_dwi, neg_bval, neg_bvec, neg_corrvols, neg_bvals, neg_tsize = self._generate_files(
            neg_keys, 'neg')

        subprocess.run(
            self.inputs.simg_cmd.run_cmd('eddy_combine').split() + [pos_dwi, pos_bval, pos_bvec,
             pos_corrvols, neg_dwi, neg_bval, neg_bvec, neg_corrvols, self.inputs.work_dir, '1'],
            check=True)
        self._results['combined_dwi_file'] = Path(self.inputs.work_dir, 'data.nii.gz')
        self._rotate_b(pos_tsize, neg_tsize, pos_bvals, neg_bvals)

        return runtime


class _WBDilateInputSpec(BaseInterfaceInputSpec):
    data_file = traits.File(mandatory=True, exists=True, desc='filename of input data')
    res = traits.Int(mandatory=True, desc='dilate resolution')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')
    simg_cmd = traits.Any(mandatory=True, desc='command for using singularity image (or not)')


class _WBDilateOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, desc='dilated data output')


class WBDilate(SimpleInterface):
    """Dilate DWI data"""
    input_spec = _WBDilateInputSpec
    output_spec = _WBDilateOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = Path(self.inputs.work_dir, 'data_dilated.nii.gz')
        args = (
            f'-volume-dilate {self.inputs.data_file} {self.inputs.res * 4} NEAREST '
            f'{self._results["out_file"]}')
        wb = workbench.base.WBCommand(command=self.inputs.simg_cmd.run_cmd('wb_command'), args=args)
        wb.run()

        return runtime


class _DilateMaskInputSpec(BaseInterfaceInputSpec):
    mask_file = traits.File(mandatory=True, exists=True, desc='filename of input mask')
    simg_cmd = traits.Any(mandatory=True, desc='command for using singularity image (or not)')


class _DilateMaskOutputSpec(TraitedSpec):
    dil0_file = traits.File(exists=True, desc='mask dilated once')
    out_file = traits.File(exists=True, desc='dilated mask output')


class DilateMask(SimpleInterface):
    """Dilate mask image 7 times"""
    input_spec = _DilateMaskInputSpec
    output_spec = _DilateMaskOutputSpec

    def _run_interface(self, runtime):
        args = '-kernel 3D -dilM'

        resamp_start = fsl.ImageMaths(
            command=self.inputs.simg_cmd.run_cmd('fslmaths'), in_file=self.inputs.mask_file,
            args=args)
        resamp_start.run()
        self._results['dil0_file'] = resamp_start.aggregate_outputs().out_file
        mask_prev = self._results['dil0_file']

        for _ in range(6):
            resamp_curr = fsl.ImageMaths(
                command=self.inputs.simg_cmd.run_cmd('fslmaths'), in_file=mask_prev, args=args)
            resamp_curr.run()
            mask_prev = resamp_curr.aggregate_outputs().out_file

        self._results['out_file'] = mask_prev

        return runtime


class _RotateBVec2StrInputSpec(BaseInterfaceInputSpec):
    bvecs_file = traits.File(mandatory=True, exists=True, desc='filename of input (merged) bvecs')
    rot = traits.List(dtype=list, mandatory=True, desc='rotation matrix of diff2str warp')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')


class _RotateBVec2StrOutputSpec(TraitedSpec):
    rotated_file = traits.File(exists=True, desc='rotated bvecs')


class RotateBVec2Str(SimpleInterface):
    """Rotate bvecs based on diffusion-to-structural warp"""
    input_spec = _RotateBVec2StrInputSpec
    output_spec = _RotateBVec2StrOutputSpec

    def _run_interface(self, runtime):
        bvecs = pd.read_csv(self.inputs.bvecs_file, delim_whitespace=True, header=None)
        rotated_bvecs = np.matmul(np.array(self.inputs.rot)[:3, :3], bvecs)
        self._results['rotated_file'] = Path(self.inputs.work_dir, 'rotated2str.bvec')
        pd.DataFrame(rotated_bvecs).to_csv(
            self._results['rotated_file'], sep=' ', header=False, index=False,
            float_format='%10.6f', quoting=3, escapechar=' ')

        return runtime


class _HCPMinProcInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(mandatory=True, desc='name of dataset (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    subject = traits.Str(mandatory=True, desc='subject ID')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')
    simg_cmd = traits.Any(mandatory=True, desc='command for using singularity image (or not)')


class _HCPMinProcOutputSpec(TraitedSpec):
    hcp_proc_wf = traits.Any(desc='HCP Minimal Processing workflow')


class HCPMinProc(SimpleInterface):
    """HCP Minimal Processing Pipeline Diffusion Processing"""
    input_spec = _HCPMinProcInputSpec
    output_spec = _HCPMinProcOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.dataset == 'HCP-A':
            dirs_phases = [('dirs', [98, 99]), ('phase', ['AP', 'PA'])]
            ds_folder = 'hcp_aging'
        elif self.inputs.dataset == 'HCP-D':
            dirs_phases = [('dirs', [98, 99]), ('phase', ['AP', 'PA'])]
            ds_folder = 'hcp_development'
        else:
            raise DatasetError()

        tmp_dir = Path(self.inputs.work_dir, 'hcp_proc_tmp')
        tmp_dir.mkdir(parents=True, exist_ok=True)
        fs_dir_guess = Path(
            self.inputs.work_dir, self.inputs.subject, 'original', 'hcp', ds_folder,
            self.inputs.subject, 'T1w')

        self._results['hcp_proc_wf'] = pe.Workflow(
            'hcp_min_proc_wf', base_dir=self.inputs.work_dir)
        inputnode = pe.Node(niu.IdentityInterface(
            fields=['d_files', 't1_files', 'fs_files', 'fs_dir']), name='inputnode')
        outputnode = pe.Node(
            niu.IdentityInterface(fields=['data', 'bval', 'bvec', 'mask']), name='outputnode')
        split_files = pe.Node(
            niu.Function(function=d_files_dirsphase, output_names=['image', 'bval', 'bvec']),
            name='split_files', iterables=dirs_phases)
        self._results['hcp_proc_wf'].connect([(inputnode, split_files, [('d_files', 'd_files')])])

        # 1. PreEddy
        # 1.1. normalize intensity
        mean_dwi = pe.Node(
            fsl.ImageMaths(
                command=self.inputs.simg_cmd.run_cmd('fslmaths'), args='-Xmean -Ymean -Zmean'),
            name='mean_dwi')
        extract_b0s = pe.Node(
            ExtractB0(dataset=self.inputs.dataset, work_dir=tmp_dir, simg_cmd=self.inputs.simg_cmd),
            name='extract_b0s')
        merge_b0s = pe.Node(
            fsl.Merge(command=self.inputs.simg_cmd.run_cmd('fslmerge'), dimension='t'),
            name='merge_b0s')
        mean_b0 = pe.Node(
            fsl.ImageMaths(command=self.inputs.simg_cmd.run_cmd('fslmaths'), args='-Tmean'),
            name='mean_b0')
        scale = pe.Node(
            fsl.ImageMeants(command=self.inputs.simg_cmd.run_cmd('fslmeants')), name='scale')
        rescale = pe.JoinNode(
            Rescale(dataset=self.inputs.dataset, simg_cmd=self.inputs.simg_cmd), name='rescale',
            joinfield=['scale_files'], joinsource='split_files')

        self._results['hcp_proc_wf'].connect([
            (split_files, mean_dwi, [('image', 'in_file')]),
            (split_files, extract_b0s, [('bval', 'bval_file')]),
            (mean_dwi, extract_b0s, [('out_file', 'dwi_file')]),
            (extract_b0s, merge_b0s, [('roi_files', 'in_files')]),
            (merge_b0s, mean_b0, [('merged_file', 'in_file')]),
            (mean_b0, scale, [('out_file', 'in_file')]),
            (inputnode, rescale, [('d_files', 'd_files')]),
            (scale, rescale, [('out_file', 'scale_files')])])

        # 1.2. prepare b0s and index files for topup
        update_rescaled = pe.Node(
            niu.Function(function=update_d_files, output_names=['d_files']), name='update_rescaled')
        update_rescaled.inputs.dataset = self.inputs.dataset
        split_rescaled = pe.Node(
            niu.Function(function=d_files_dirsphase, output_names=['image', 'bval', 'bvec']),
            name='split_rescaled', iterables=dirs_phases)
        rescaled_b0s = pe.Node(
            ExtractB0(
                dataset=self.inputs.dataset, work_dir=tmp_dir, rescale=True,
                simg_cmd=self.inputs.simg_cmd),
            name='rescaled_b0s')
        b0_list = pe.JoinNode(
            niu.Function(function=flatten_list, output_names=['out_list']), name='b0_list',
            joinfield='in_list', joinsource='split_rescaled')
        pos_b0_list = pe.JoinNode(
            niu.Function(function=flatten_list, output_names=['out_list']), name='pos_b0_list',
            joinfield='in_list', joinsource='split_rescaled')
        neg_b0_list = pe.JoinNode(
            niu.Function(function=flatten_list, output_names=['out_list']), name='neg_b0_list',
            joinfield='in_list', joinsource='split_rescaled')
        merge_rescaled_b0s = pe.Node(
            fsl.Merge(command=self.inputs.simg_cmd.run_cmd('fslmerge'), dimension='t'),
            name='merge_rescaled_b0s')
        merge_pos_b0s = pe.Node(
            fsl.Merge(command=self.inputs.simg_cmd.run_cmd('fslmerge'), dimension='t'),
            name='merge_pos_b0s')
        merge_neg_b0s = pe.Node(
            fsl.Merge(command=self.inputs.simg_cmd.run_cmd('fslmerge'), dimension='t'),
            name='merge_neg_b0s')

        self._results['hcp_proc_wf'].connect([
            (inputnode, update_rescaled, [('d_files', 'd_files')]),
            (rescale, update_rescaled, [('rescaled_files', 'dwi_replacements')]),
            (update_rescaled, split_rescaled, [('d_files', 'd_files')]),
            (split_rescaled, rescaled_b0s, [('bval', 'bval_file'), ('image', 'dwi_file')]),
            (rescaled_b0s, b0_list, [('roi_files', 'in_list')]),
            (rescaled_b0s, pos_b0_list, [('pos_files', 'in_list')]),
            (rescaled_b0s, neg_b0_list, [('neg_files', 'in_list')]),
            (b0_list, merge_rescaled_b0s, [('out_list', 'in_files')]),
            (pos_b0_list, merge_pos_b0s, [('out_list', 'in_files')]),
            (neg_b0_list, merge_neg_b0s, [('out_list', 'in_files')])])

        # 1.3. topup
        topup_config_file = Path(tmp_dir, 'HCP_pipeline', 'global', 'config', 'b02b0.cnf')
        if not topup_config_file.is_file():
            dl.clone(
                'git@github.com:Washington-University/HCPpipelines.git',
                path=Path(tmp_dir, 'HCP_pipeline'))
        prepare_topup = pe.Node(PrepareTopup(dataset=self.inputs.dataset), name='prepare_topup')
        estimate_topup = pe.Node(
            fsl.TOPUP(command=self.inputs.simg_cmd.run_cmd('topup'), config=str(topup_config_file)),
            name='estimate_topup')
        pos_b01 = pe.Node(
            fsl.ExtractROI(command=self.inputs.simg_cmd.run_cmd('fslroi'), t_min=0, t_size=1),
            name='pos_b01')
        neg_b01 = pe.Node(
            fsl.ExtractROI(command=self.inputs.simg_cmd.run_cmd('fslroi'), t_min=0, t_size=1),
            name='neg_b01')
        b01_files = pe.Node(
            niu.Function(function=create_2item_list, output_names=['out_list']), name='b01_files')
        apply_topup = pe.Node(
            fsl.ApplyTOPUP(command=self.inputs.simg_cmd.run_cmd('applytopup'), method='jac'),
            name='apply_topup')
        nodif_brainmask = pe.Node(
            niu.Function(function=bet_nodif_mask, output_names=['mask_file']),
            name='nodif_brainmask')
        nodif_brainmask.inputs.run_cmd = self.inputs.simg_cmd.run_cmd('bet')
        #nodif_brainmask.inputs.work_dir = str(Path(
        #    self.inputs.work_dir, 'hcp_min_proc_wf', 'nodif_brainmask'))
        nodif_brainmask.inputs.work_dir = str(tmp_dir)

        self._results['hcp_proc_wf'].connect([
            (update_rescaled, prepare_topup, [('d_files', 'd_files')]),
            (b0_list, prepare_topup, [('out_list', 'roi_files')]),
            (merge_pos_b0s, prepare_topup, [('merged_file', 'pos_b0_file')]),
            (merge_rescaled_b0s, estimate_topup, [('merged_file', 'in_file')]),
            (prepare_topup, estimate_topup, [
                ('enc_dir', 'encoding_direction'), ('ro_time', 'readout_times')]),
            (merge_pos_b0s, pos_b01, [('merged_file', 'in_file')]),
            (merge_neg_b0s, neg_b01, [('merged_file', 'in_file')]),
            (pos_b01, b01_files, [('roi_file', 'item1')]),
            (neg_b01, b01_files, [('roi_file', 'item2')]),
            (prepare_topup, apply_topup, [('indices_t', 'in_index')]),
            (estimate_topup, apply_topup, [
                ('out_enc_file', 'encoding_file'), ('out_fieldcoef', 'in_topup_fieldcoef'),
                ('out_movpar', 'in_topup_movpar')]),
            (b01_files, apply_topup, [('out_list', 'in_files')]),
            (apply_topup, nodif_brainmask, [('out_corrected', 'in_file')])])

        # 2. Eddy
        split_files_type = pe.Node(
            niu.Function(function=d_files_type, output_names=['image', 'bval', 'bvec']),
            name='split_files_type')
        merge_bfiles = pe.Node(
            MergeBFiles(dataset=self.inputs.dataset, work_dir=tmp_dir), name='merge_bfiles')
        merge_rescaled_dwi = pe.Node(
            fsl.Merge(command=self.inputs.simg_cmd.run_cmd('fslmerge'), dimension='t'),
            name='merge_rescaled_dwi')
        eddy_index = pe.Node(
            EddyIndex(dataset=self.inputs.dataset, work_dir=tmp_dir), name='eddy_index')
        eddy = pe.Node(
            fsl.Eddy(command=self.inputs.simg_cmd.run_cmd('eddy'), fwhm=0, args='-v'), name='eddy')

        self._results['hcp_proc_wf'].connect([
            (update_rescaled, split_files_type, [('d_files', 'd_files')]),
            (split_files_type, merge_bfiles, [('bval', 'bval_files'), ('bvec', 'bvec_files')]),
            (split_files_type, merge_rescaled_dwi, [('image', 'in_files')]),
            (b0_list, eddy_index, [('out_list', 'roi_files')]),
            (split_files_type, eddy_index, [('image', 'dwi_files')]),
            (merge_rescaled_dwi, eddy, [('merged_file', 'in_file')]),
            (merge_bfiles, eddy, [('bval_merged', 'in_bval'), ('bvec_merged', 'in_bvec')]),
            (estimate_topup, eddy, [('out_enc_file', 'in_acqp')]),
            (eddy_index, eddy, [('index_file', 'in_index')]),
            (nodif_brainmask, eddy, [('mask_file', 'in_mask')]),
            (estimate_topup, eddy, [
                ('out_fieldcoef', 'in_topup_fieldcoef'), ('out_movpar', 'in_topup_movpar')])])

        # 3. PostEddy
        # 3.1. postproc
        eddy_postproc = pe.Node(
            EddyPostProc(dataset=self.inputs.dataset, work_dir=tmp_dir,
                         simg_cmd=self.inputs.simg_cmd), name='eddy_postproc')
        fov_mask = pe.Node(
            fsl.ImageMaths(
                command=self.inputs.simg_cmd.run_cmd('fslmaths'), args='-abs -Tmin -bin -fillh'),
            name='fov_mask')
        mask_to_args = pe.Node(
            niu.Function(function=combine_2strings, output_names=['out_str']), name='mask_to_args')
        mask_to_args.inputs.str1 = '-mas '
        mask_data = pe.Node(
            fsl.ImageMaths(command=self.inputs.simg_cmd.run_cmd('fslmaths')), name='mask_data')
        thresh_data = pe.Node(
            fsl.ImageMaths(command=self.inputs.simg_cmd.run_cmd('fslmaths'), args='-thr 0'),
            name='thresh_data')

        self._results['hcp_proc_wf'].connect([
            (split_files_type, eddy_postproc, [
                ('bval', 'bval_files'), ('bvec', 'bvec_files'), ('image', 'rescaled_files')]),
            (eddy, eddy_postproc, [
                ('out_corrected', 'eddy_corrected_file'),
                ('out_rotated_bvecs', 'eddy_bvecs_file')]),
            (eddy_postproc, outputnode, [('rot_bvals', 'bval')]),
            (eddy_postproc, fov_mask, [('combined_dwi_file', 'in_file')]),
            (fov_mask, mask_to_args, [('out_file', 'str2')]),
            (eddy_postproc, mask_data, [('combined_dwi_file', 'in_file')]),
            (mask_to_args, mask_data, [('out_str', 'args')]),
            (mask_data, thresh_data, [('out_file', 'in_file')])
        ])

        # 3.2. DiffusionToStructural
        nodif_brain = pe.Node(
            fsl.ExtractROI(command=self.inputs.simg_cmd.run_cmd('fslroi'), t_min=0, t_size=1),
            name='nodif_brain')
        split_t1_files = pe.Node(
            niu.Function(function=t1_files_type, output_names=[
                't1', 't1_restore', 't1_restore_brain', 'bias', 'fs_mask', 'xfm']),
            name='split_t1_file')
        wm_seg = pe.Node(
            fsl.FAST(command=self.inputs.simg_cmd.run_cmd('fast'), output_type='NIFTI_GZ'),
            name='wm_seg')
        pve_file = pe.Node(
            niu.Function(
                function=last_list_item, input_names=['in_list'], output_names=['out_item']),
            name='pve_file')
        wm_thresh = pe.Node(
            fsl.ImageMaths(command=self.inputs.simg_cmd.run_cmd('fslmaths'), args='-thr 0.5 -bin'),
            name='wm_thresh')
        flirt_init = pe.Node(
            fsl.FLIRT(command=self.inputs.simg_cmd.run_cmd('flirt'), dof=6), name='flirt_init')
        if self.inputs.simg_cmd.cmd is None:
            schedule_file = Path(getenv('FSLDIR'), 'etc', 'flirtsch', 'bbr.sch')
        else: # guess path in the container
            schedule_file = '/usr/local/fsl/etc/flirtsch/bbr.sch'
        flirt_nodif2t1 = pe.Node(
            niu.Function(function=flirt_bbr_sch, output_names=['out_matrix_file']),
            name='flirt_nodif2t1')
        flirt_nodif2t1.inputs.run_cmd = self.inputs.simg_cmd.run_cmd('flirt')
        flirt_nodif2t1.inputs.sch_file = str(schedule_file)
        flirt_nodif2t1.inputs.out_dir = str(tmp_dir)
        flirt_nodif2t1.inputs.out_prefix = 'nodif2t1'
        nodif_t1 = pe.Node(
            fsl.ApplyWarp(command=self.inputs.simg_cmd.run_cmd('applywarp'), interp='spline',
                          relwarp=True), name='nodif_t1')
        bias_to_args = pe.Node(
            niu.Function(function=combine_2strings, output_names=['out_str']), name='bias_to_args')
        bias_to_args.inputs.str1 = '-div '
        nodif_bias = pe.Node(
            fsl.ImageMaths(command=self.inputs.simg_cmd.run_cmd('fslmaths')), name='nodif_bias')
        split_fs_files = pe.Node(
            niu.Function(function=fs_files_type, output_names=[
                'l_whited', 'r_whited', 'eye', 'orig', 'l_thick', 'r_thick']),
            name='split_fs_files')
        bbr_epi2t1 = pe.Node(
            freesurfer.BBRegister(
                command=self.inputs.simg_cmd.run_cmd(
                    'bbregister', options=f'--env SUBJECTS_DIR={fs_dir_guess}'),
                contrast_type='bold', dof=6, args='--surf white.deformed',
                subject_id=self.inputs.subject),
            name='bbr_epi2t1')
        # Due to a nipype bug, fsl_out must be supplied with a file for now
        fsl_file = Path(tmp_dir, 'tkr_diff2str.mat')
        tkr_diff2str = pe.Node(
            freesurfer.Tkregister2(command=self.inputs.simg_cmd.run_cmd('tkregister2'),
                                   noedit=True, fsl_out=fsl_file), name='tkr_diff2str')
        diff2str = pe.Node(
            fsl.ConvertXFM(command=self.inputs.simg_cmd.run_cmd('convert_xfm'), concat_xfm=True),
            name='diff2str')
        res = pe.Node(niu.Function(function=diff_res, output_names=['res', 'dilate']), name='res')
        flirt_resamp = pe.Node(
            fsl.FLIRT(command=self.inputs.simg_cmd.run_cmd('flirt')), name='flirt_resampe')
        t1_resamp = pe.Node(
            fsl.ApplyWarp(command=self.inputs.simg_cmd.run_cmd('applywarp'), interp='spline',
                          relwarp=True), name='t1_resamp')
        dilate_data = pe.Node(
            WBDilate(work_dir=tmp_dir, simg_cmd=self.inputs.simg_cmd), name='dilate_data')
        resamp_data = pe.Node(
            fsl.FLIRT(command=self.inputs.simg_cmd.run_cmd('flirt'), apply_xfm=True,
                      interp='spline'), name='resamp_data')
        mask_resamp = pe.Node(
            fsl.FLIRT(command=self.inputs.simg_cmd.run_cmd('flirt'), interp='nearestneighbour'),
            name='mask_resamp')
        mask_dilate = pe.Node(DilateMask(simg_cmd=self.inputs.simg_cmd), name='mask_dilate')
        fmask_t1 = pe.Node(
            fsl.FLIRT(command=self.inputs.simg_cmd.run_cmd('flirt'), apply_xfm=True,
                      interp='trilinear'), name='fmask_resamp')
        fmask_thresh = pe.Node(
            fsl.ImageMaths(command=self.inputs.simg_cmd.run_cmd('fslmaths'),
                           args='-thr 0.999 -bin'),
            name='fmask_thresh')
        masks_to_args = pe.Node(
            niu.Function(function=combine_4strings, output_names=['out_str']), name='masks_to_args')
        masks_to_args.inputs.str1 = '-mas '
        masks_to_args.inputs.str3 = ' -mas '
        fmask_data = pe.Node(
            fsl.ImageMaths(command=self.inputs.simg_cmd.run_cmd('fslmaths')), name='fmask_data')
        nonneg_data = pe.Node(
            fsl.ImageMaths(command=self.inputs.simg_cmd.run_cmd('fslmaths'), args='-thr 0'),
            name='nonneg_data')
        mask_mean = pe.Node(
            fsl.ImageMaths(command=self.inputs.simg_cmd.run_cmd('fslmaths'), args='-Tmean'),
            name='mask_mean')
        mean_to_args = pe.Node(
            niu.Function(function=combine_2strings, output_names=['out_str']), name='mean_to_args')
        mean_to_args.inputs.str1 = '-mas '
        mask_mask = pe.Node(
            fsl.ImageMaths(command=self.inputs.simg_cmd.run_cmd('fslmaths')), name='mask_mask')
        rot_matrix = pe.Node(
            fsl.AvScale(command=self.inputs.simg_cmd.run_cmd('avscale')), name='rot_matrix')
        rotate_bvec = pe.Node(RotateBVec2Str(work_dir=tmp_dir), name='rotate_bvec')

        self._results['hcp_proc_wf'].connect([
            (thresh_data, nodif_brain, [('out_file', 'in_file')]),
            (inputnode, split_t1_files, [('t1_files', 't1_files')]),
            (split_t1_files, wm_seg, [('t1_restore_brain', 'in_files')]),
            (wm_seg, pve_file, [('partial_volume_files', 'in_list')]),
            (pve_file, wm_thresh, [('out_item', 'in_file')]),
            (nodif_brain, flirt_init, [('roi_file', 'in_file')]),
            (split_t1_files, flirt_init, [('t1_restore_brain', 'reference')]),
            (nodif_brain, flirt_nodif2t1, [('roi_file', 'in_file')]),
            (split_t1_files, flirt_nodif2t1, [('t1', 'reference')]),
            (wm_thresh, flirt_nodif2t1, [('out_file', 'wm_seg')]),
            (flirt_init, flirt_nodif2t1, [('out_matrix_file', 'in_matrix_file')]),
            (nodif_brain, nodif_t1, [('roi_file', 'in_file')]),
            (split_t1_files, nodif_t1, [('t1', 'ref_file')]),
            (flirt_nodif2t1, nodif_t1, [('out_matrix_file', 'premat')]),
            (split_t1_files, bias_to_args, [('bias', 'str2')]),
            (nodif_t1, nodif_bias, [('out_file', 'in_file')]),
            (bias_to_args, nodif_bias, [('out_str', 'args')]),
            (inputnode, split_fs_files, [('fs_files', 'fs_files')]),
            (inputnode, bbr_epi2t1, [('fs_dir', 'subjects_dir')]),
            (nodif_bias, bbr_epi2t1, [('out_file', 'source_file')]),
            (split_fs_files, bbr_epi2t1, [('eye', 'init_reg_file')]),
            (nodif_bias, tkr_diff2str, [('out_file', 'moving_image')]),
            (bbr_epi2t1, tkr_diff2str, [('out_reg_file', 'reg_file')]),
            (split_t1_files, tkr_diff2str, [('t1', 'target_image')]),
            (flirt_nodif2t1, diff2str, [('out_matrix_file', 'in_file')]),
            (tkr_diff2str, diff2str, [('fsl_file', 'in_file2')]),
            (thresh_data, res, [('out_file', 'data_file')]),
            (split_t1_files, flirt_resamp, [
                ('t1_restore', 'in_file'), ('t1_restore', 'reference')]),
            (res, flirt_resamp, [('res', 'apply_isoxfm')]),
            (split_t1_files, t1_resamp, [('t1_restore', 'in_file')]),
            (flirt_resamp, t1_resamp, [('out_file', 'ref_file')]),
            (thresh_data, dilate_data, [('out_file', 'data_file')]),
            (res, dilate_data, [('dilate', 'res')]),
            (dilate_data, resamp_data, [('out_file', 'in_file')]),
            (t1_resamp, resamp_data, [('out_file', 'reference')]),
            (diff2str, resamp_data, [('out_file', 'in_matrix_file')]),
            (split_t1_files, mask_resamp, [('fs_mask', 'in_file'), ('fs_mask', 'reference')]),
            (res, mask_resamp, [('res', 'apply_isoxfm')]),
            (mask_resamp, mask_dilate, [('out_file', 'mask_file')]),
            (fov_mask, fmask_t1, [('out_file', 'in_file')]),
            (t1_resamp, fmask_t1, [('out_file', 'reference')]),
            (diff2str, fmask_t1, [('out_file', 'in_matrix_file')]),
            (fmask_t1, fmask_thresh, [('out_file', 'in_file')]),
            (mask_dilate, masks_to_args, [('out_file', 'str2')]),
            (fmask_thresh, masks_to_args, [('out_file', 'str4')]),
            (resamp_data, fmask_data, [('out_file', 'in_file')]),
            (masks_to_args, fmask_data, [('out_str', 'args')]),
            (fmask_data, nonneg_data, [('out_file', 'in_file')]),
            (nonneg_data, mask_mean, [('out_file', 'in_file')]),
            (mask_mean, mean_to_args, [('out_file', 'str2')]),
            (mask_dilate, mask_mask, [('dil0_file', 'in_file')]),
            (mean_to_args, mask_mask, [('out_str', 'args')]),
            (diff2str, rot_matrix, [('out_file', 'mat_file')]),
            (eddy_postproc, rotate_bvec, [('rot_bvecs', 'bvecs_file')]),
            (rot_matrix, rotate_bvec, [('rotation_translation_matrix', 'rot')]),
            (nonneg_data, outputnode, [('out_file', 'data')]),
            (mask_mask, outputnode, [('out_file', 'mask')]),
            (rotate_bvec, outputnode, [('rotated_file', 'bvec')])])

        return runtime


class _CSDInputSpec(BaseInterfaceInputSpec):
    data = traits.File(mandatory=True, exists=True, desc='DWI data file')
    bval = traits.File(mandatory=True, exists=True, desc='b value file')
    bvec = traits.File(mandatory=True, exists=True, desc='b vector files')
    mask = traits.File(mandatory=True, exists=True, desc='mask file')
    dataset = traits.Str(
        mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')
    simg_cmd = traits.Any(mandatory=True, desc='command for using singularity image (or not)')


class _CSDOutputSpec(TraitedSpec):
    fod_wm_file = traits.File(exists=True, desc='white matter FOD file')


class CSD(SimpleInterface):
    """Fiber orientation distribution estimation using Constrained Spherical Deconvolution (CSD)"""
    input_spec = _CSDInputSpec
    output_spec = _CSDOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            shells = '1500,3000'
        elif self.inputs.dataset == 'HCP-YA':
            shells = '1000, 2000, 3000'
        else:
            raise DatasetError()

        res_wm_file = Path(self.inputs.work_dir, 'sfwm.txt')
        res_gm_file = Path(self.inputs.work_dir, 'gm.txt')
        res_csf_file = Path(self.inputs.work_dir, 'csf.txt')
        subprocess.run(
            self.inputs.simg_cmd.run_cmd('dwi2response').split() + ['dhollander', '-shells', shells,
             '-nthreads', '0', '-force', '-mask', str(self.inputs.mask),
             '-fslgrad', str(self.inputs.bvec), str(self.inputs.bval), str(self.inputs.data),
             str(res_wm_file), str(res_gm_file), str(res_csf_file)], check=True)

        self._results['fod_wm_file'] = Path(self.inputs.work_dir, 'fod_wm.mif')
        fod_gm_file = Path(self.inputs.work_dir, 'fod_gm.mif')
        fod_csf_file = Path(self.inputs.work_dir, 'fod_csf.mif')
        fod = self.inputs.simg_cmd.run_cmd('dwi2fod').split() + [
            'msmt_csd', '-shells', shells,
            '-nthreads', '0', '-mask', str(self.inputs.mask),
            '-fslgrad', str(self.inputs.bvec), str(self.inputs.bval), str(self.inputs.data),
            str(res_wm_file), str(self._results['fod_wm_file']),
            str(res_gm_file), str(fod_gm_file),
            str(res_csf_file), str(fod_csf_file)]
        if not self._results['fod_wm_file'].is_file():
            subprocess.run(fod, check=True)

        return runtime


class _TCKInputSpec(BaseInterfaceInputSpec):
    fod_wm_file = traits.File(mandatory=True, exists=True, desc='white matter FOD file')
    fs_dir = traits.Directory(desc='FreeSurfer subject directory')
    dataset = traits.Str(
        mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    bval = traits.File(mandatory=True, exists=True, desc='b value file')
    bvec = traits.File(mandatory=True, exists=True, desc='b vector files')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')

    simg_cmd = traits.Any(mandatory=True, desc='command for using singularity image (or not)')


class _TCKOutputSpec(TraitedSpec):
    tck_file = traits.File(exists=True, desc='tracks file')


class TCK(SimpleInterface):
    """Probabilistic Tractography"""
    input_spec = _TCKInputSpec
    output_spec = _TCKOutputSpec

    def _params(self) -> dict:
        voxel_sizes = {'HCP-YA': 1.25, 'HCP-A': 1.5, 'HCP-D': 1.5, 'ABCD': 1.7, 'UKB': 2}
        voxel_size = voxel_sizes[self.inputs.dataset]
        params = {
            # default parameters of tckgen
            'algorithm': 'iFOD2', 'step': str(0.5 * voxel_size), 'angle': '45',
            'minlength': str(2 * voxel_size), 'cutoff': '0.05', 'trials': '1000', 'samples': '4',
            'downsample': '3', 'power': '0.33',
            # parameters different from default (Jung et al. 2021)
            'maxlength': '250', 'max_attempts_per_seed': '50', 'select': '10000000'}

        return params

    def _run_interface(self, runtime):
        ftt_file = Path(self.inputs.work_dir, 'ftt.nii.gz')
        ftt = self.inputs.simg_cmd.run_cmd('5ttgen').split() + [
            'hsvs', str(self.inputs.fs_dir),str(ftt_file)]
        if not ftt_file.is_file():
            subprocess.run(ftt, check=True)

        seed_file = Path(self.inputs.work_dir, 'WBT_10M_seeds_ctx.txt')
        self._results['tck_file'] = Path(self.inputs.work_dir, 'WBT_10M_ctx.tck')
        params = self._params()
        tck = self.inputs.simg_cmd.run_cmd('tckgen').split()
        for param, value in params.items():
            tck = tck + [f'-{param}', value]
        tck = tck + [
            '-seed_dynamic', str(self.inputs.fod_wm_file), '-act', str(ftt_file),
            '-output_seeds', str(seed_file), '-backtrack', '-crop_at_gmwmi', '-nthreads', '0',
            '-fslgrad', str(self.inputs.bvec), str(self.inputs.bval),
            str(self.inputs.fod_wm_file), str(self._results['tck_file'])]
        if not self._results['tck_file'].is_file():
            subprocess.run(tck, check=True)

        return runtime
