from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
from pathlib import Path
import pandas as pd
import nibabel as nib
import subprocess
import numpy as np

from nipype.interfaces import fsl, workbench

# functions to help bridge nipype interfaces

def d_files_dirsphase(d_files, dirs, phase):
    key = f'dir{dirs}_{phase}'
    image = d_files[f'{key}.nii.gz']
    bval = d_files[f'{key}.bval']
    bvec = d_files[f'{key}.bvec']

    return image, bval, bvec

def d_files_type(d_files):
    image = [d_files[key] for key in d_files if '.nii.gz' in key]
    bval = [d_files[key] for key in d_files if '.bval' in key]
    bvec = [d_files[key] for key in d_files if '.bvec' in key]

    return image, bval, bvec

def t1_files(t1_files):
    t1_file = t1_files['t1']
    t1_restore_file = t1_files['t1_restore']
    t1_brain_file = t1_files['t1_restore_brain']
    bias_file = t1_files['bias']
    mask_file = t1_files['fs_mask']
    t1_to_mni = t1_files['t1_to_mni']

    return t1_file, t1_restore_file, t1_brain_file, bias_file, mask_file, t1_to_mni

def fs_files(fs_files):
    from pathlib import Path

    lh_whitedeform_file = fs_files['lh_white_deformed']
    rh_whitedeform_file = fs_files['rh_white_deformed']
    eye_file = fs_files['eye']
    orig_file = fs_files['orig']
    lh_thick_file = fs_files['lh_thickness']
    rh_thick_file = fs_files['rh_thickness']
    subdir = Path(fs_files['orig']).parent.parent.parent

    return lh_whitedeform_file, rh_whitedeform_file, eye_file, orig_file, lh_thick_file, rh_thick_file, subdir

def fs_files_aparc(fs_files):
    lh_aparc = fs_files['lh_aparc']
    rh_aparc = fs_files['rh_aparc']
    lh_white = fs_files['lh_white']
    rh_white = fs_files['rh_white']
    lh_pial = fs_files['lh_pial']
    rh_pial = fs_files['rh_pial']
    lh_ribbon = fs_files['lh_ribbon']
    rh_ribbon = fs_files['rh_ribbon']
    ribbon = fs_files['ribbon']

    return lh_aparc, rh_aparc, lh_white, rh_white, lh_pial, rh_pial, lh_ribbon, rh_ribbon, ribbon

def update_d_files(d_files, dataset, dwi_replacements):
    if dataset == 'HCP-A' or dataset == 'HCP-D':
        keys = ['dir98_AP', 'dir98_PA', 'dir99_AP', 'dir99_PA']
    for key in keys:
        dwi_key = [d_key for d_key in d_files if key in d_key and '.nii.gz' in d_key]
        dwi_replace = [d_file for d_file in dwi_replacements if key in str(d_file)]
        d_files[dwi_key[0]] = dwi_replace[0]

    return d_files

def flatten_list(in_list):
    import itertools
    return list(itertools.chain.from_iterable(in_list))

def create_2item_list(item1, item2):
    return [item1, item2]

def combine_2strings(str1, str2):
    return f'{str1}{str2}'

def combine_4strings(str1, str2, str3, str4):
    return f'{str1}{str2}{str3}{str4}'

def last_list_item(in_list):
    return in_list[-1]

def diff_res(data_file):
    import nibabel as nib
    res = nib.load(data_file).header.get_zooms()[0]

    return res, int(res*4)

# extract b0 slices

class _ExtractB0InputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    bval_file = traits.File(mandatory=True, desc='absolute path to the bval file')
    dwi_file = traits.File(mandatory=True, desc='absolute path to the DWI file')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')
    rescale = traits.Bool(False, desc='if b0dist should be applied on rescaled data')

class _ExtractB0OutputSpec(TraitedSpec):
    roi_files = traits.List(dtpye=Path, desc='filenames of B0 files')
    pos_files = traits.List(dtpye=Path, desc='filenames of B0 files with positive phase encoding')
    neg_files = traits.List(dtpye=Path, desc='filenames of B0 files with negative phase encoding')

class ExtractB0(SimpleInterface):
    input_spec = _ExtractB0InputSpec
    output_spec = _ExtractB0OutputSpec

    def _extract_b0(self, b0dist=None):
        b0maxbval = 50 # values below this will be considered as b0s
        bvals = pd.read_csv(self.inputs.bval_file, header=None, delim_whitespace=True).squeeze('rows')

        if b0dist is None:
            dist_count = 0
            roi_files = [self.inputs.dwi_file]
        else:
            dist_count = b0dist + 1
            roi_files = []
        dim4 = nib.load(self.inputs.dwi_file).header.get_data_shape()[3]
        vol_count = 0

        for b in bvals:
            roi_file = Path(self.inputs.work_dir, f'roi{vol_count}_{Path(self.inputs.dwi_file).name}')
            if b < b0maxbval and b0dist is None:
                roi = fsl.ExtractROI(in_file=self.inputs.dwi_file, t_min=dist_count, t_size=1, roi_file=roi_file)
                roi.run()
                roi_files.append(roi_file)
            elif b < b0maxbval and vol_count < dim4 and dist_count > b0dist:
                roi = fsl.ExtractROI(in_file=self.inputs.dwi_file, t_min=vol_count, t_size=1,
                                     roi_file=roi_file)
                roi.run()
                roi_files.append(roi_file)
                dist_count = 0
            dist_count = dist_count + 1
            vol_count = vol_count + 1

        return roi_files

    def _split_pos_neg(self, roi_files):
        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            pos_files = [roi_file for roi_file in roi_files if 'AP' in str(roi_file)]
            neg_files = [roi_file for roi_file in roi_files if 'PA' in str(roi_file)]

        return pos_files, neg_files

    def _run_interface(self, runtime):
        if not self.inputs.rescale:
            self._results['roi_files'] = self._extract_b0()
        else:
            b0dist = 45 # minimum distance between b0s
            self._results['roi_files'] = self._extract_b0(b0dist=b0dist)
            self._results['pos_files'], self._results['neg_files'] = self._split_pos_neg(self._results['roi_files'])

        return runtime
    
# rescale DWI files (except the first one)

class _RescaleInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    scale_files = traits.List(mandatory=True, dtype=str, desc='filenames of scale files')
    d_files = traits.Dict(mandatory=True, dtype=Path, desc='filenames of diffusion data')

class _RescaleOutputSpec(TraitedSpec):
    rescaled_files = traits.List(dtype=str, desc='filenames of rescaled DWI images')

class Rescale(SimpleInterface):
    input_spec = _RescaleInputSpec
    output_spec = _RescaleOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            key = 'dir98_AP'
            keys = ['dir98_PA', 'dir99_AP', 'dir99_PA']

        rescale_file = [s_file for s_file in self.inputs.scale_files if key in s_file]
        rescale = pd.read_csv(rescale_file[0], header=None).squeeze()
        self._results['rescaled_files'] = [self.inputs.d_files[d_key] for d_key in self.inputs.d_files if key in d_key]

        for key in keys:
            scale_file = [s_file for s_file in self.inputs.scale_files if key in s_file]
            scale = pd.read_csv(scale_file[0], header=None).squeeze()
            d_file = [self.inputs.d_files[d_key] for d_key in self.inputs.d_files if key in d_key]
            maths = fsl.ImageMaths(in_file=d_file[0], args=f'-mul {rescale} -div {scale}')
            maths.run()
            self._results['rescaled_files'].append(maths.aggregate_outputs().out_file)

        return runtime
    
# prepare for FSL Topup (susceptibility distortion correction)

class _PrepareTopupInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    roi_files = traits.List(mandatory=True, dtpye=Path, desc='filenames of B0 files')
    d_files = traits.Dict(mandatory=True, dtype=Path, desc='filenames of diffusion data')
    pos_b0_file = traits.File(mandatory=True, desc='merged positive b0 file')

class _PrepareTopupOutputSpec(TraitedSpec):
    enc_dir = traits.List(dtype=str, desc='encoding directions for each b0')
    ro_time = traits.Float(desc='readout time')
    indices_t = traits.List(dtype=int, desc='indices based on time dimension of b0 files')

class PrepareTopup(SimpleInterface):
    input_spec = _PrepareTopupInputSpec
    output_spec = _PrepareTopupOutputSpec

    def _encoding_direction(self):
        enc_dir = []
        for roi_file in self.inputs.roi_files:
            if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
                if 'AP' in str(roi_file):
                    enc_dir.append('y')
                elif 'PA' in str(roi_file):
                    enc_dir.append('y-')

        return enc_dir
    
    def _readout_time(self):
        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            dim_p = nib.load(self.inputs.d_files['dir98_AP.nii.gz']).header.get_data_shape()[1] # dim2 for AP/PA encoding
            echospacing = 0.69 # based on protocol files in 'HCP_VE11C_Prisma_2019.01.14' from HCP Lifespan
        ro_time = round(echospacing * (dim_p - 1) / 1000, 6)

        return ro_time
    
    def _dim_t(self):
        return [1, nib.load(self.inputs.pos_b0_file).header.get_data_shape()[3] + 1]

    def _run_interface(self, runtime):
        self._results['enc_dir'] = self._encoding_direction()
        self._results['ro_time'] = self._readout_time()
        self._results['indices_t'] = self._dim_t()

        return runtime
    
# merge bvals and bvecs

class _MergeBFilesInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    bval_files = traits.List(mandatory=True, dtype=Path, desc='list of bval files to merge')
    bvec_files = traits.List(mandatory=True, dtype=Path, desc='list of bvec files to merge')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')

class _MergeBFilesOutputSpec(TraitedSpec):
    bval_merged = traits.File(exists=True, desc='merged bval file')
    bvec_merged = traits.File(exists=True, desc='merged bvec file')

class MergeBFiles(SimpleInterface):
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
                bvals = pd.concat([bvals, pd.read_csv(bval_file[0], delim_whitespace=True, header=None)], axis=1)
                bvecs = pd.concat([bvecs, pd.read_csv(bvec_file[0], delim_whitespace=True, header=None)], axis=1)

        self._results['bval_merged'] = Path(self.inputs.work_dir, 'merged.bval')
        self._results['bvec_merged'] = Path(self.inputs.work_dir, 'merged.bvec')
        bvals.to_csv(self._results['bval_merged'], sep='\t', header=None, index=None)
        bvecs.to_csv(self._results['bvec_merged'], sep='\t', header=None, index=None)

        return runtime
    
# create index file for eddy correction
class _EddyIndexInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    roi_files = traits.List(mandatory=True, dtype=Path, desc='filenames of B0 files')
    dwi_files = traits.List(mandatory=True, dtype=Path, desc='filenames of rescaled DWI images')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')

class _EddyIndexOutputSpec(TraitedSpec):
    index_file = traits.File(exists=True, desc='filename of index file')

class EddyIndex(SimpleInterface):
    input_spec = _EddyIndexInputSpec
    output_spec = _EddyIndexOutputSpec

    def _run_interface(self, runtime):
        rois = [int(str(roi_file.name).lstrip('roi').split('_')[0]) for roi_file in self.inputs.roi_files]
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

        self._results['index_file'] = Path(self.inputs.work_dir, 'index.txt')
        pd.DataFrame(indices).to_csv(self._results['index_file'], sep='\t', header=None, index=None)

        return runtime

# combine output files and rotate bvals/bvecs from eddy correction

class _EddyPostProcInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    bval_files = traits.List(mandatory=True, dtype=Path, desc='list of bval files to merge')
    bvec_files = traits.List(mandatory=True, dtype=Path, desc='list of bvec files to merge')
    eddy_corrected_file = traits.File(mandatory=True, exists=True, desc='filename of eddy corrected image')
    eddy_bvecs_file = traits.File(mandatory=True, exists=True, desc='filename of eddy corrected bvecs')
    rescaled_files = traits.List(mandatory=True, dtype=Path, desc='filenames of rescaled DWI images')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')

class _EddyPostProcOutputSpec(TraitedSpec):
    combined_dwi_file = traits.File(exists=True, desc='combined DWI data')
    rot_bvals = traits.File(exists=True, desc='average rotated bvals')
    rot_bvecs = traits.File(exists=True, desc='average rotated bvecs')

class EddyPostProc(SimpleInterface):
    input_spec = _EddyPostProcInputSpec
    output_spec = _EddyPostProcOutputSpec

    def _generate_files(self, keys, dirs):
        bvals = pd.DataFrame()
        bvecs = pd.DataFrame()
        corrvols = []
        tsizes = []
        for key in keys:
            bval_file = [b_file for b_file in self.inputs.bval_files if key in str(b_file)]
            bvec_file = [b_file for b_file in self.inputs.bvec_files if key in str(b_file)]
            bval = pd.read_csv(bval_file[0], delim_whitespace=True, header=None)
            bvals = pd.concat([bvals, bval], axis=1)
            bvecs = pd.concat([bvecs, pd.read_csv(bvec_file[0], delim_whitespace=True, header=None)], axis=1)

            rescaled_file = [d_file for d_file in self.inputs.rescaled_files if key in str(d_file)]
            dim4 = nib.load(rescaled_file[0]).header.get_data_shape()[3]
            corrvols.append([dim4, dim4])
            tsizes.append(bval.shape[1])

        bval_merged = Path(self.inputs.work_dir, f'{dirs}.bval')
        bvec_merged = Path(self.inputs.work_dir, f'{dirs}.bvec')
        bvals.to_csv(bval_merged, sep='\t', header=None, index=None)
        bvecs.to_csv(bvec_merged, sep='\t', header=None, index=None)

        corrvols_file = Path(self.inputs.work_dir, f'{dirs}_volnum.txt')
        pd.DataFrame(corrvols).to_csv(corrvols_file, sep='\t', header=None, index=None)

        bval_tsize = bvals.shape[1]
        extract_roi = fsl.ExtractROI(in_file=self.inputs.eddy_corrected_file, t_size=bval_tsize)

        if dirs == 'pos':
            extract_roi.inputs.t_min = 0
        elif dirs == 'neg':
            extract_roi.inputs.t_min = bval_tsize
        extract_roi.run()

        return extract_roi.aggregate_outputs().roi_file, bval_merged, bvec_merged, corrvols_file, bvals, tsizes
    
    def _rotate_b(self, pos_tsize, neg_tsize, pos_bvals, neg_bvals):
        rot_bvecs = pd.read_csv(self.inputs.eddy_bvecs_file, delim_whitespace=True, header=None)
        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            pos_rot_bvecs = np.zeros((3, sum(pos_tsize)))
            neg_rot_bvecs = np.zeros((3, sum(neg_tsize)))
            break_pos = [pos_tsize[0], pos_tsize[0]+neg_tsize[0], sum(pos_tsize)+neg_tsize[0]]
            pos_rot_bvecs[:, :pos_tsize[0]] = rot_bvecs.iloc[:, :break_pos[0]]
            neg_rot_bvecs[:, :neg_tsize[0]] = rot_bvecs.iloc[:, break_pos[0]:break_pos[1]]
            pos_rot_bvecs[:, pos_tsize[0]:] = rot_bvecs.iloc[:, break_pos[1]:break_pos[2]]
            neg_rot_bvecs[:, neg_tsize[0]:] = rot_bvecs.iloc[:, break_pos[2]:]

            avg_bvals = np.zeros((sum(pos_tsize)), dtype='i4')
            avg_bvecs = np.zeros((3, sum(pos_tsize)))
            for i in range(sum(pos_tsize)):
                pos_bvec = np.array(pos_bvals.iloc[:, i]) * np.array(pos_rot_bvecs[:, i]).reshape((3, 1))
                neg_bvec = np.array(neg_bvals.iloc[:, i]) * np.array(neg_rot_bvecs[:, i]).reshape((3, 1))
                bvec_sum = (np.dot(pos_bvec, pos_bvec.T) + np.dot(neg_bvec, neg_bvec.T)) / 2
                eigvals, eigvecs = np.linalg.eig(bvec_sum)
                eigvalmax = np.argmax(eigvals)
                avg_bvals[i] = np.rint(eigvals[eigvalmax] ** 0.5)
                avg_bvecs[:, i] = eigvecs[:, eigvalmax]
            
            self._results['rot_bvals'] = Path(self.inputs.work_dir, 'rotated.bval')
            self._results['rot_bvecs'] = Path(self.inputs.work_dir, 'rotated.bvec')
            pd.DataFrame(avg_bvals).T.to_csv(self._results['rot_bvals'], sep=' ', header=None, index=None)
            pd.DataFrame(avg_bvecs).to_csv(self._results['rot_bvecs'], sep=' ', header=None, index=None, 
                                           float_format='%0.16f')
            
            return

    def _run_interface(self, runtime):
        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            pos_keys = ['dir98_AP', 'dir99_AP']
            neg_keys = ['dir98_PA', 'dir99_PA']
        pos_dwi, pos_bval, pos_bvec, pos_corrvols, pos_bvals, pos_tsize = self._generate_files(pos_keys, 'pos')
        neg_dwi, neg_bval, neg_bvec, neg_corrvols, neg_bvals, neg_tsize = self._generate_files(neg_keys, 'neg')

        combine_command = ['eddy_combine', pos_dwi, pos_bval, pos_bvec, pos_corrvols,
                           neg_dwi, neg_bval, neg_bvec, neg_corrvols,
                           self.inputs.work_dir, '1']
        subprocess.run(combine_command, check=True)
        self._results['combined_dwi_file'] = Path(self.inputs.work_dir, 'data.nii.gz')

        self._rotate_b(pos_tsize, neg_tsize, pos_bvals, neg_bvals)

        return runtime

# dilate data using Connectome Workbench

class _WBDilateInputSpec(BaseInterfaceInputSpec):
    data_file = traits.File(mandatory=True, exists=True, desc='filename of input data')
    res = traits.Int(mandatory=True, desc='dilate resolution')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')

class _WBDilateOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, desc='dilated data output')

class WBDilate(SimpleInterface):
    input_spec = _WBDilateInputSpec
    output_spec = _WBDilateOutputSpec

    def _run_interface(self, runtime):
            self._results['out_file'] = Path(self.inputs.work_dir, 'data_dilated.nii.gz')
            args = f'-volume-dilate {self.inputs.data_file} {self.inputs.res*4} NEAREST {self._results["out_file"]}'
            wb = workbench.base.WBCommand(command='wb_command', args=args)
            wb.run()

            return runtime
    
# dilate mask multiple times

class _DilateMaskInputSpec(BaseInterfaceInputSpec):
    mask_file = traits.File(mandatory=True, exists=True, desc='filename of input mask')

class _DilateMaskOutputSpec(TraitedSpec):
    dil0_file = traits.File(exists=True, desc='mask dilated once')
    out_file = traits.File(exists=True, desc='dilated mask output')

class DilateMask(SimpleInterface):
    input_spec = _DilateMaskInputSpec
    output_spec = _DilateMaskOutputSpec

    def _run_interface(self, runtime):
            args='-kernel 3D -dilM'

            resamp_start = fsl.ImageMaths(in_file=self.inputs.mask_file, args=args)
            resamp_start.run()
            self._results['dil0_file'] = resamp_start.aggregate_outputs().out_file
            mask_prev = self._results['dil0_file']

            for _ in range(6):
                resamp_curr = fsl.ImageMaths(in_file=mask_prev, args=args)
                resamp_curr.run()
                mask_prev = resamp_curr.aggregate_outputs().out_file

            self._results['out_file'] = mask_prev

            return runtime
    
# rotate bvecs based on diffusion-to-structural warp

class _RotateBVec2StrInputSpec(BaseInterfaceInputSpec):
    bvecs_file = traits.File(mandatory=True, exists=True, desc='filename of input (merged) bvecs')
    rot = traits.List(dtype=list, mandatory=True, desc='rotation matrix of diff2str warp')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')

class _RotateBVec2StrOutputSpec(TraitedSpec):
    rotated_file = traits.File(exists=True, desc='rotated bvecs')

class RotateBVec2Str(SimpleInterface):
    input_spec = _RotateBVec2StrInputSpec
    output_spec = _RotateBVec2StrOutputSpec

    def _run_interface(self, runtime):
            bvecs = pd.read_csv(self.inputs.bvecs_file, delim_whitespace=True, header=None)
            rotated_bvecs = np.matmul(np.array(self.inputs.rot)[:3, :3], bvecs)
            self._results['rotated_file'] = Path(self.inputs.work_dir, 'rotated2str.bvec')
            pd.DataFrame(rotated_bvecs).to_csv(self._results['rotated_file'], sep=' ', header=None, index=None,
                                               float_format='%10.6f', quoting=3, escapechar=' ')

            return runtime