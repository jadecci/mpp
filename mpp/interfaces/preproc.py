from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import nipype.pipeline as pe
from nipype.interfaces import utility as niu
from pathlib import Path
import datalad.api as dl
from os import getenv
import subprocess
import logging

from nipype.interfaces import fsl, freesurfer

from mpp.utilities.preproc import (d_files_dirsphase, d_files_type, t1_files, fs_files, update_d_files,
                                   flatten_list, create_2item_list, last_list_item,
                                   combine_2strings, combine_4strings, diff_res,
                                   ExtractB0, Rescale, PrepareTopup, MergeBFiles, EddyIndex, 
                                   EddyPostProc, WBDilate, DilateMask, RotateBVec2Str)

logging.getLogger('datalad').setLevel(logging.WARNING)

### HCPMinProc: HCP Minimal Processing Pipeline

class _HCPMinProcInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(mandatory=True, desc='name of dataset (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    subject = traits.Str(mandatory=True, desc='subject ID')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')

class _HCPMinProcOutputSpec(TraitedSpec):
    hcp_proc_wf = traits.Any(desc='HCP Minimal Processing workflow')

class HCPMinProc(SimpleInterface):
    input_spec = _HCPMinProcInputSpec
    output_spec = _HCPMinProcOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            dirs_phases = [('dirs', [98, 99]), ('phase', ['AP', 'PA'])]
            
        tmp_dir = Path(self.inputs.work_dir, 'hcp_proc_tmp')
        tmp_dir.mkdir(parents=True, exist_ok=True)

        self._results['hcp_proc_wf'] = pe.Workflow('hcp_min_proc_wf', base_dir=self.inputs.work_dir)
        inputnode = pe.Node(niu.IdentityInterface(fields=['d_files', 'fs_files']), name='inputnode')
        outputnode = pe.Node(niu.IdentityInterface(fields=['data', 'bval', 'bvec', 'mask']), name='outputnode')
        split_files = pe.Node(niu.Function(function=d_files_dirsphase, output_names=['image', 'bval', 'bvec']),
                              name='split_files', iterables=dirs_phases)
        self._results['hcp_proc_wf'].connect([(inputnode, split_files, [('d_files', 'd_files')])])
        
        # 1. PreEddy
        # 1.1. normalize intensity
        mean_dwi = pe.Node(fsl.ImageMaths(args='-Xmean -Ymean -Zmean'), name='mean_dwi')
        extract_b0s = pe.Node(ExtractB0(dataset=self.inputs.dataset, work_dir=tmp_dir), name='extract_b0s')
        merge_b0s = pe.Node(fsl.Merge(dimension='t'), name='merge_b0s')
        mean_b0 = pe.Node(fsl.ImageMaths(args='-Tmean'), name='mean_b0')
        scale = pe.Node(fsl.ImageMeants(), name='scale')
        rescale = pe.JoinNode(Rescale(dataset=self.inputs.dataset), name='rescale', joinfield=['scale_files'], joinsource='split_files')

        self._results['hcp_proc_wf'].connect([(split_files, mean_dwi, [('image', 'in_file')]),
                                              (split_files, extract_b0s, [('bval', 'bval_file')]),
                                              (mean_dwi, extract_b0s, [('out_file', 'dwi_file')]),
                                              (extract_b0s, merge_b0s, [('roi_files', 'in_files')]),
                                              (merge_b0s, mean_b0, [('merged_file', 'in_file')]),
                                              (mean_b0, scale, [('out_file', 'in_file')]),
                                              (inputnode, rescale, [('d_files', 'd_files')]),
                                              (scale, rescale, [('out_file', 'scale_files')])])
        
        # 1.2. prepare b0s and index files for topup
        update_rescaled = pe.Node(niu.Function(function=update_d_files, output_names=['d_files']),
                                  name='update_rescaled')
        update_rescaled.inputs.dataset = self.inputs.dataset
        split_rescaled = pe.Node(niu.Function(function=d_files_dirsphase, output_names=['image', 'bval', 'bvec']), 
                                 name='split_rescaled', iterables=dirs_phases)
        rescaled_b0s = pe.Node(ExtractB0(dataset=self.inputs.dataset, work_dir=tmp_dir, rescale=True), 
                               name='rescaled_b0s')
        b0_list = pe.JoinNode(niu.Function(function=flatten_list, output_names=['out_list']), name='b0_list',
                              joinfield='in_list', joinsource='split_rescaled')
        pos_b0_list = pe.JoinNode(niu.Function(function=flatten_list, output_names=['out_list']), name='pos_b0_list',
                                  joinfield='in_list', joinsource='split_rescaled')
        neg_b0_list = pe.JoinNode(niu.Function(function=flatten_list, output_names=['out_list']), name='neg_b0_list',
                                  joinfield='in_list', joinsource='split_rescaled')
        merge_rescaled_b0s = pe.Node(fsl.Merge(dimension='t'), name='merge_rescaled_b0s')
        merge_pos_b0s = pe.Node(fsl.Merge(dimension='t'), name='merge_pos_b0s')
        merge_neg_b0s = pe.Node(fsl.Merge(dimension='t'), name='merge_neg_b0s')
        
        self._results['hcp_proc_wf'].connect([(inputnode, update_rescaled, [('d_files', 'd_files')]),
                                              (rescale, update_rescaled, [('rescaled_files', 'dwi_replacements')]),
                                              (update_rescaled, split_rescaled, [('d_files', 'd_files')]),
                                              (split_rescaled, rescaled_b0s, [('bval', 'bval_file'),
                                                                              ('image', 'dwi_file')]),
                                              (rescaled_b0s, b0_list, [('roi_files', 'in_list')]),
                                              (rescaled_b0s, pos_b0_list, [('pos_files', 'in_list')]),
                                              (rescaled_b0s, neg_b0_list, [('neg_files', 'in_list')]),
                                              (b0_list, merge_rescaled_b0s, [('out_list', 'in_files')]),
                                              (pos_b0_list, merge_pos_b0s, [('out_list', 'in_files')]),
                                              (neg_b0_list, merge_neg_b0s, [('out_list', 'in_files')])])
        
        # 1.3. topup
        topup_config_file = Path(tmp_dir, 'HCP_pipeline', 'global', 'config', 'b02b0.cnf')
        if not topup_config_file.is_file():
            dl.clone('git@github.com:Washington-University/HCPpipelines.git', path=Path(tmp_dir, 'HCP_pipeline'))
        prepare_topup = pe.Node(PrepareTopup(dataset=self.inputs.dataset), name='prepare_topup')
        estimate_topup = pe.Node(fsl.TOPUP(config=str(topup_config_file)), name='estimate_topup')
        pos_b01 = pe.Node(fsl.ExtractROI(t_min=0, t_size=1), name='pos_b01')
        neg_b01 = pe.Node(fsl.ExtractROI(t_min=0, t_size=1), name='neg_b01')
        b01_files = pe.Node(niu.Function(function=create_2item_list, output_names=['out_list']), name='b01_files')
        apply_topup = pe.Node(fsl.ApplyTOPUP(method='jac'), name='apply_topup')
        nodif_brainmask = pe.Node(fsl.BET(mask=True, frac=0.2), name='nodif_brainmask')

        self._results['hcp_proc_wf'].connect([(update_rescaled, prepare_topup, [('d_files', 'd_files')]),
                                              (b0_list, prepare_topup, [('out_list', 'roi_files')]),
                                              (merge_pos_b0s, prepare_topup, [('merged_file', 'pos_b0_file')]),
                                              (merge_rescaled_b0s, estimate_topup, [('merged_file', 'in_file')]),
                                              (prepare_topup, estimate_topup, [('enc_dir', 'encoding_direction'),
                                                                               ('ro_time', 'readout_times')]),
                                              (merge_pos_b0s, pos_b01, [('merged_file', 'in_file')]),
                                              (merge_neg_b0s, neg_b01, [('merged_file', 'in_file')]),
                                              (pos_b01, b01_files, [('roi_file', 'item1')]),
                                              (neg_b01, b01_files, [('roi_file', 'item2')]),
                                              (prepare_topup, apply_topup, [('indices_t', 'in_index')]),
                                              (estimate_topup, apply_topup, [('out_enc_file', 'encoding_file'),
                                                                             ('out_fieldcoef', 'in_topup_fieldcoef'),
                                                                             ('out_movpar', 'in_topup_movpar')]),
                                              (b01_files, apply_topup, [('out_list', 'in_files')]),
                                              (apply_topup, nodif_brainmask, [('out_corrected', 'in_file')])])

        # 2. Eddy
        split_files_type = pe.Node(niu.Function(function=d_files_type, output_names=['image', 'bval', 'bvec']),
                                   name='split_files_type')
        merge_bfiles = pe.Node(MergeBFiles(dataset=self.inputs.dataset, work_dir=tmp_dir), name='merge_bfiles')
        merge_rescaled_dwi = pe.Node(fsl.Merge(dimension='t'), name='merge_rescaled_dwi')
        eddy_index = pe.Node(EddyIndex(dataset=self.inputs.dataset, work_dir=tmp_dir), name='eddy_index')
        eddy = pe.Node(fsl.Eddy(fwhm=0, args='-v'), name='eddy')
        
        self._results['hcp_proc_wf'].connect([(update_rescaled, split_files_type, [('d_files', 'd_files')]),
                                              (split_files_type, merge_bfiles, [('bval', 'bval_files'),
                                                                                ('bvec', 'bvec_files')]),
                                              (split_files_type, merge_rescaled_dwi, [('image', 'in_files')]),
                                              (b0_list, eddy_index, [('out_list', 'roi_files')]),
                                              (split_files_type, eddy_index, [('image', 'dwi_files')]),
                                              (merge_rescaled_dwi, eddy, [('merged_file', 'in_file')]),
                                              (merge_bfiles, eddy, [('bval_merged', 'in_bval'),
                                                                    ('bvec_merged', 'in_bvec')]),
                                              (estimate_topup, eddy, [('out_enc_file', 'in_acqp')]),
                                              (eddy_index, eddy, [('index_file', 'in_index')]),
                                              (nodif_brainmask, eddy, [('mask_file', 'in_mask')]),
                                              (estimate_topup, eddy, [('out_fieldcoef', 'in_topup_fieldcoef'),
                                                                      ('out_movpar', 'in_topup_movpar')])])
        
        # 3. PostEddy
        # 3.1. postproc
        eddy_postproc = pe.Node(EddyPostProc(dataset=self.inputs.dataset, work_dir=tmp_dir), name='eddy_postproc')
        fov_mask = pe.Node(fsl.ImageMaths(args='-abs -Tmin -bin -fillh'), name='fov_mask')
        mask_to_args = pe.Node(niu.Function(function=combine_2strings, output_names=['out_str']), name='mask_to_args')
        mask_to_args.inputs.str1 = '-mas '
        mask_data = pe.Node(fsl.ImageMaths(), name='mask_data')
        thresh_data = pe.Node(fsl.ImageMaths(args='-thr 0'), name='thresh_data')

        self._results['hcp_proc_wf'].connect([(split_files_type, eddy_postproc, [('bval', 'bval_files'),
                                                                                 ('bvec', 'bvec_files'),
                                                                                 ('image', 'rescaled_files')]),
                                              (eddy, eddy_postproc, [('out_corrected', 'eddy_corrected_file'),
                                                                     ('out_rotated_bvecs', 'eddy_bvecs_file')]),
                                              (eddy_postproc, outputnode, [('rot_bvals', 'bval')]),
                                              (eddy_postproc, fov_mask, [('combined_dwi_file', 'in_file')]),
                                              (fov_mask, mask_to_args, [('out_file', 'str2')]),
                                              (eddy_postproc, mask_data, [('combined_dwi_file', 'in_file')]),
                                              (mask_to_args, mask_data, [('out_str', 'args')]),
                                              (mask_data, thresh_data, [('out_file', 'in_file')])])
        
        # 3.2. DiffusionToStructural
        nodif_brain = pe.Node(fsl.ExtractROI(t_min=0, t_size=1), name='nodif_brain')
        split_t1_files = pe.Node(niu.Function(function=t1_files, output_names=['t1', 't1_restore', 't1_restore_brain',
                                                                               'bias', 'fs_mask', 'xfm']),
                                 name='split_t1_file')
        wm_seg = pe.Node(fsl.FAST(), name='wm_seg')
        pve_file = pe.Node(niu.Function(function=last_list_item, input_names=['in_list'], output_names=['out_item']),
                           name='pve_file')
        wm_thresh = pe.Node(fsl.ImageMaths(args='-thr 0.5 -bin'), name='wm_thresh')
        flirt_init = pe.Node(fsl.FLIRT(dof=6), name='flirt_init')
        schedule_file = Path(getenv('FSLDIR'), 'etc', 'flirtsch', 'bbr.sch')
        flirt_nodif2t1 = pe.Node(fsl.FLIRT(dof=6, cost='bbr', schedule=schedule_file), name='flirt_nodif2t1')
        nodif_t1 = pe.Node(fsl.ApplyWarp(interp='spline', relwarp=True), name='nodif_t1')
        bias_to_args = pe.Node(niu.Function(function=combine_2strings, output_names=['out_str']), name='bias_to_args')
        bias_to_args.inputs.str1 = '-div '
        nodif_bias = pe.Node(fsl.ImageMaths(), name='nodif_bias')
        split_fs_files = pe.Node(niu.Function(function=fs_files, output_names=['l_whited', 'r_whited', 'eye', 'orig',
                                                                               'l_thick', 'r_thick', 'subdir']), 
                                 name='split_fs_files')
        bbr_epi2t1 = pe.Node(freesurfer.BBRegister(contrast_type='bold', dof=6, args='--surf white.deformed',
                                                   subject_id=self.inputs.subject),
                             name='bbr_epi2t1')
        tkr_diff2str = pe.Node(freesurfer.Tkregister2(noedit=True), name='tkr_diff2str')
        diff2str = pe.Node(fsl.ConvertXFM(concat_xfm=True), name='diff2str')
        res = pe.Node(niu.Function(function=diff_res, output_names=['res', 'dilate']), name='res')
        flirt_resamp = pe.Node(fsl.FLIRT(), name='flirt_resampe')
        t1_resamp = pe.Node(fsl.ApplyWarp(interp='spline', relwarp=True), name='t1_resamp')
        dilate_data = pe.Node(WBDilate(work_dir=tmp_dir), name='dilate_data')
        resamp_data = pe.Node(fsl.FLIRT(apply_xfm=True, interp='spline'), name='resamp_data')
        mask_resamp = pe.Node(fsl.FLIRT(interp='nearestneighbour'), name='mask_resamp')
        mask_dilate = pe.Node(DilateMask(), name='mask_dilate')
        fmask_t1 = pe.Node(fsl.FLIRT(apply_xfm=True, interp='trilinear'), name='fmask_resamp')
        fmask_thresh = pe.Node(fsl.ImageMaths(args='-thr 0.999 -bin'), name='fmask_thresh')
        masks_to_args = pe.Node(niu.Function(function=combine_4strings, output_names=['out_str']), name='masks_to_args')
        masks_to_args.inputs.str1 = '-mas '
        masks_to_args.inputs.str3 = ' -mas '
        fmask_data = pe.Node(fsl.ImageMaths(), name='fmask_data')
        nonneg_data = pe.Node(fsl.ImageMaths(args='-thr 0'), name='nonneg_data')
        mask_mean = pe.Node(fsl.ImageMaths(args='-Tmean'), name='mask_mean')
        mean_to_args = pe.Node(niu.Function(function=combine_2strings, output_names=['out_str']), name='mean_to_args')
        mean_to_args.inputs.str1 = '-mas '
        mask_mask = pe.Node(fsl.ImageMaths(), name='mask_mask')
        rot_matrix = pe.Node(fsl.AvScale(), name='rot_matrix')
        rotate_bvec = pe.Node(RotateBVec2Str(work_dir=tmp_dir), name='rotate_bvec')

        self._results['hcp_proc_wf'].connect([(thresh_data, nodif_brain, [('out_file', 'in_file')]),
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
                                              (nodif_bias, bbr_epi2t1, [('out_file', 'source_file')]),
                                              (split_fs_files, bbr_epi2t1, [('subdir', 'subjects_dir'),
                                                                            ('eye', 'init_reg_file')]),
                                              (nodif_bias, tkr_diff2str, [('out_file', 'moving_image')]),
                                              (bbr_epi2t1, tkr_diff2str, [('out_reg_file', 'reg_file')]),
                                              (split_t1_files, tkr_diff2str, [('t1', 'target_image')]),
                                              (flirt_nodif2t1, diff2str, [('out_matrix_file', 'in_file')]),
                                              (tkr_diff2str, diff2str, [('fsl_file', 'in_file2')]),
                                              (thresh_data, res, [('out_file', 'data_file')]),
                                              (split_t1_files, flirt_resamp, [('t1_restore', 'in_file'),
                                                                              ('t1_restore', 'reference')]),
                                              (res, flirt_resamp, [('res', 'apply_isoxfm')]),
                                              (split_t1_files, t1_resamp, [('t1_restore', 'in_file')]),
                                              (flirt_resamp, t1_resamp, [('out_file', 'ref_file')]),
                                              (thresh_data, dilate_data, [('out_file', 'data_file')]),
                                              (res, dilate_data, [('dilate', 'res')]),
                                              (dilate_data, resamp_data, [('out_file', 'in_file')]),
                                              (t1_resamp, resamp_data, [('out_file', 'reference')]),
                                              (diff2str, resamp_data, [('out_file', 'in_matrix_file')]),
                                              (split_t1_files, mask_resamp, [('fs_mask', 'in_file'),
                                                                             ('fs_mask', 'reference')]),
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
    
### CSD: fiber orientation distribution estimation

class _CSDInputSpec(BaseInterfaceInputSpec):
    data = traits.File(mandatory=True, exists=True, desc='DWI data file')
    bval = traits.File(mandatory=True, exists=True, desc='b value file')
    bvec = traits.File(mandatory=True, exists=True, desc='b vector files')
    mask = traits.File(mandatory=True, exists=True, desc='mask file')
    dataset = traits.Str(mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')

class _CSDOutputSpec(TraitedSpec):
    fod_wm_file = traits.File(exists=True, desc='white matter FOD file')

class CSD(SimpleInterface):
    input_spec = _CSDInputSpec
    output_spec = _CSDOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            shells = '1500,3000'

        res_wm_file = Path(self.inputs.work_dir, 'sfwm.txt')
        res_gm_file = Path(self.inputs.work_dir, 'gm.txt')
        res_csf_file = Path(self.inputs.work_dir, 'csf.txt')
        response = ['dwi2response', 'dhollander', '-shells', shells, '-nthreads', '0', '-force',
                    '-mask', str(self.inputs.mask), 
                    '-fslgrad', str(self.inputs.bvec), str(self.inputs.bval), str(self.inputs.data),
                    str(res_wm_file), str(res_gm_file), str(res_csf_file)]
        subprocess.run(response, check=True)

        self._results['fod_wm_file'] = Path(self.inputs.work_dir, 'fod_wm.mif')
        fod_gm_file = Path(self.inputs.work_dir, 'fod_gm.mif')
        fod_csf_file = Path(self.inputs.work_dir, 'fod_csf.mif')
        fod = ['dwi2fod', 'msmt_csd', '-shells', shells, '-nthreads', '0',
               '-mask', str(self.inputs.mask),
               '-fslgrad', str(self.inputs.bvec), str(self.inputs.bval), str(self.inputs.data),
               str(res_wm_file), str(self._results['fod_wm_file']),
               str(res_gm_file), str(fod_gm_file),
               str(res_csf_file), str(fod_csf_file)]
        if not self._results['fod_wm_file'].is_file():
            subprocess.run(fod, check=True)

        return runtime
    
### TCK: probabilistic tractography

class _TCKInputSpec(BaseInterfaceInputSpec):
    fod_wm_file = traits.File(mandatory=True, exists=True, desc='white matter FOD file')
    fs_dir = traits.Directory(desc='FreeSurfer subject directory')
    bval = traits.File(mandatory=True, exists=True, desc='b value file')
    bvec = traits.File(mandatory=True, exists=True, desc='b vector files')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')

class _TCKOutputSpec(TraitedSpec):
    tck_file = traits.File(exists=True, desc='tracks file')

class TCK(SimpleInterface):
    input_spec = _TCKInputSpec
    output_spec = _TCKOutputSpec

    def _run_interface(self, runtime):
        # parameters (Jung et al. 2021)
        algorithm = 'iFOD2'
        step = '0.625'
        angle = '45'
        minlength = '2.5'
        maxlength = '250'
        cutoff = '0.06'
        trials = '1000'
        downsample = '3'
        max_attempts_per_seed = '50'
        tract_schaefer = '10000000'

        # default settings
        samples = '4'
        power = '0.25'

        ftt_file = Path(self.inputs.work_dir, 'ftt.nii.gz')
        ftt = ['5ttgen', 'hsvs', str(self.inputs.fs_dir), str(ftt_file)]
        if not ftt_file.is_file():
            subprocess.run(ftt, check=True)

        seed_file = Path(self.inputs.work_dir, 'WBT_10M_seeds_ctx.txt')
        self._results['tck_file'] = Path(self.inputs.work_dir, 'WBT_10M_ctx.tck')
        tck = ['tckgen', '-algorithm', algorithm, '-select', tract_schaefer, '-step', step, '-angle', angle,
               '-minlength', minlength, '-maxlength', maxlength, '-cutoff', cutoff, '-trials', trials,
               '-downsample', downsample, '-seed_dynamic', str(self.inputs.fod_wm_file),
               '-max_attempts_per_seed', max_attempts_per_seed, '-output_seeds', str(seed_file),
               '-act', str(ftt_file), '-backtrack', '-crop_at_gmwmi',
               '-samples', samples, '-power', power, '-nthreads', '0',
               '-fslgrad', str(self.inputs.bvec), str(self.inputs.bval),
               str(self.inputs.fod_wm_file), str(self._results['tck_file'])]
        if not self._results['tck_file'].is_file():
            subprocess.run(tck, check=True)
        
        return runtime
