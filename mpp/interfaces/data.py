import subprocess
from pathlib import Path
import logging

import pandas as pd
import numpy as np
from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import datalad.api as dl

from mpp.utilities.data import write_h5, pheno_hcp
from mpp.utilities.features import pheno_conf_hcp
from mpp.exceptions import DatasetError


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
            if dataset in ['HCP-A', 'HCP-D']:
                sublist = [str(file)[-19:-9] for file in features_files]
            else:
                sublist = [str(file)[-9:-3] for file in features_files]
            sublist, _ = pheno_conf_hcp(
                dataset, self.inputs.phenotypes_dir[dataset], self.inputs.features_dir[dataset],
                sublist)
            sublist, pheno, pheno_perm = pheno_hcp(
                dataset, self.inputs.phenotypes_dir[dataset], self.inputs.phenotype, sublist)
            _, self._results['confounds'] = pheno_conf_hcp(
                dataset, self.inputs.phenotypes_dir[dataset], self.inputs.features_dir[dataset],
                sublist)
            self._results['phenotypes'] = pheno
            self._results['phenotypes_perm'] = pheno_perm
            self._results['sublists'][dataset] = sublist

        return runtime


class _PredictionSaveInputSpec(BaseInterfaceInputSpec):
    results = traits.List(dtype=dict, desc='accuracy results')
    output_dir = traits.Directory(mandatory=True, desc='absolute path to output directory')
    overwrite = traits.Bool(mandatory=True, desc='whether to overwrite existing results')
    phenotype = traits.Str(mandatory=True, desc='phenotype to use as prediction target')
    type = traits.Str(mandatory=True, desc='type of model used in prediction')


class PredictionSave(SimpleInterface):
    """Save prediction results"""
    input_spec = _PredictionSaveInputSpec

    def _run_interface(self, runtime):
        Path(self.inputs.output_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(
            self.inputs.output_dir, f'{self.inputs.type}_results_{self.inputs.phenotype}.h5')
        results = {key: item for d in self.inputs.results for key, item in d.items()}
        for key, val in results.items():
            write_h5(output_file, f'/{key}', np.array(val), self.inputs.overwrite)

        return runtime
