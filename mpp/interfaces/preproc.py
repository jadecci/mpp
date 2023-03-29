from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import nipype.pipeline as pe
from nipype.interfaces import utility as niu
from pathlib import Path
import datalad.api as dl
import logging

from nipype.interfaces import fsl

from mpp.utilities.preproc import ExtractB0, Rescale, PrepareTopup

logging.getLogger('datalad').setLevel(logging.WARNING)

### HCPMinProc: HCP Minimal Processing Pipeline

class _HCPMinProcInputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(mandatory=True, desc='name of dataset (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')

class _HCPMinProcOutputSpec(TraitedSpec):
    hcp_proc_wf = traits.Any(desc='HCP Minimal Processing workflow')

class HCPMinProc(SimpleInterface):
    input_spec = _HCPMinProcInputSpec
    output_spec = _HCPMinProcOutputSpec

    def _run_interface(self, runtime):

        def _split_files(d_files, dirs, phase, split_type=False):
            if split_type:
                image = [d_files[key] for key in d_files if '.nii.gz' in key]
                bval = [d_files[key] for key in d_files if '.bval' in key]
                bvec = [d_files[key] for key in d_files if '.bvec' in key]
            else:
                key = f'dir{dirs}_{phase}'
                image = d_files[f'{key}.nii.gz']
                bval = d_files[f'{key}.bval']
                bvec = d_files[f'{key}.bvec']

            return image, bval, bvec
        
        def _update_files(d_files, dataset, dwi_replacements=None):
            if dataset == 'HCP-A' or dataset == 'HCP-D':
                keys = ['dir98_AP', 'dir98_PA', 'dir99_AP', 'dir99_PA']
            for key in keys:
                if dwi_replacements is not None:
                    dwi_key = [d_key for d_key in d_files if key in d_key and '.nii.gz' in d_key]
                    dwi_replace = [d_file for d_file in dwi_replacements if key in str(d_file)]
                    d_files[dwi_key[0]] = dwi_replace[0]

            return d_files
        
        def _flatten_list(in_list):
            import itertools
            return list(itertools.chain.from_iterable(in_list))
        
        def _create_list(item1, item2):
            return [item1, item2]

        if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
            dirs_phases = [('dirs', [98, 99]), ('phase', ['AP', 'PA'])]
            
        tmp_dir = Path(self.inputs.work_dir, 'hcp_proc_tmp')
        tmp_dir.mkdir(parents=True, exist_ok=True)

        self._results['hcp_proc_wf'] = pe.Workflow('hcp_min_proc_wf', base_dir=self.inputs.work_dir)
        inputnode = pe.Node(niu.IdentityInterface(fields=['dataset_dir', 'd_files', 'fs_files']), name='inputnode')
        split_files = pe.Node(niu.Function(function=_split_files, 
                                           output_names=['image', 'bval', 'bvec']), 
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
        update_rescaled = pe.Node(niu.Function(function=_update_files, output_names=['d_files']),
                                  name='update_rescaled')
        update_rescaled.inputs.dataset = self.inputs.dataset
        split_rescaled = pe.Node(niu.Function(function=_split_files, output_names=['image', 'bval', 'bvec']), 
                                 name='split_rescaled', iterables=dirs_phases)
        rescaled_b0s = pe.Node(ExtractB0(dataset=self.inputs.dataset, work_dir=tmp_dir, rescale=True), 
                               name='rescaled_b0s')
        b0_list = pe.JoinNode(niu.Function(function=_flatten_list, output_names=['out_list']), name='b0_list',
                              joinfield='in_list', joinsource='split_rescaled')
        pos_b0_list = pe.JoinNode(niu.Function(function=_flatten_list, output_names=['out_list']), name='pos_b0_list',
                                  joinfield='in_list', joinsource='split_rescaled')
        neg_b0_list = pe.JoinNode(niu.Function(function=_flatten_list, output_names=['out_list']), name='neg_b0_list',
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
        b01_files = pe.Node(niu.Function(function=_create_list, output_names=['out_list']), name='b01_files')
        apply_topup = pe.Node(fsl.ApplyTOPUP(method='jac'), name='apply_topup')
        nodif_brain = pe.Node(fsl.BET(mask=True, frac=0.2), name='nodif_brain')

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
                                              (apply_topup, nodif_brain, [('out_corrected', 'in_file')])])

        # 2. Eddy
        split_files_type = pe.Node(niu.Function(function=_split_files, output_names=['image', 'bval', 'bvec']),
                                   name='split_files_type')
        split_files_type.inputs.split_type = True
        
        self._results['hcp_proc_wf'].connect([(update_rescaled, split_files_type, [('d_files', 'd_files')])])

        return runtime