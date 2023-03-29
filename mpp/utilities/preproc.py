from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
from pathlib import Path
import pandas as pd
import nibabel as nib

from nipype.interfaces import fsl

# extract b0 slices

class _ExtractB0InputSpec(BaseInterfaceInputSpec):
    dataset = traits.Str(mandatory=True, desc='name of dataset to get (HCP-YA, HCP-A, HCP-D, ABCD, UKB)')
    bval_file = traits.File(mandatory=True, desc='absolute path to the bval file')
    dwi_file = traits.File(mandatory=True, desc='absolute path to the DWI file')
    work_dir = traits.Directory(mandatory=True, desc='absolute path to work directory')
    rescale = traits.Bool(False, desc='if b0dist should be applied on rescaled data')

class _ExtractB0OutputSpec(TraitedSpec):
    roi_files = traits.List(desc='filenames of B0 files')
    pos_files = traits.List(desc='filenames of B0 files with positive phase encoding')
    neg_files = traits.List(desc='filenames of B0 files with negative phase encoding')

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
            roi_file = Path(self.inputs.work_dir, f'roi{dist_count}_{Path(self.inputs.dwi_file).name}')
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
    
    def _create_index_file(self, roi_files):
        indices = []
        pos_count = 0
        neg_count = 0
        for roi_file in roi_files:
            if self.inputs.dataset == 'HCP-A' or self.inputs.dataset == 'HCP-D':
                if 'AP' in str(roi_file):
                    pos_count = pos_count + 1
                    indices.append(pos_count)
                elif 'PA' in str(roi_file):
                    neg_count = neg_count + 1
                    indices.append(neg_count)

        index_file = Path(self.inputs.work_dir, 'index.txt')
        pd.DataFrame(indices).to_csv(index_file, sep='\t', header=None, index=None)

        return

    def _run_interface(self, runtime):
        if not self.inputs.rescale:
            self._results['roi_files'] = self._extract_b0()
        else:
            b0dist = 45 # minimum distance between b0s
            self._results['roi_files'] = self._extract_b0(b0dist=b0dist)
            self._results['pos_files'], self._results['neg_files'] = self._split_pos_neg(self._results['roi_files'])
            self._create_index_file(self._results['roi_files'])

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
    roi_files = traits.List(mandatory=True, desc='filenames of B0 files')
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