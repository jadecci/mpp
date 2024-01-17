from pathlib import Path

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import pandas as pd
import numpy as np

from mpp.exceptions import DatasetError


class _PredictSublistInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    target = traits.Str(mandatory=True, desc="target phenotype to predict")


class _PredictSublistOutputSpec(TraitedSpec):
    sublists = traits.Dict(dtype=list, desc="availabel subjects in each dataset")


class PredictSublist(SimpleInterface):
    """Extract available phenotype data and required list of subjects"""
    input_spec = _PredictSublistInputSpec
    output_spec = _PredictSublistOutputSpec

    def _check_sub(self, subject_file: Path) -> bool:
        pheno = pd.DataFrame(pd.read_hdf(subject_file, "phenotype"))
        conf = pd.DataFrame(pd.read_hdf(subject_file, "confound"))
        if pheno[self.inputs.target].isnull().values[0]:
            return False
        if conf.isnull().values.any():
            return False

    def _run_interface(self, runtime):
        self._results["sublists"] = {}
        for dataset in self.inputs.config["datasets"]:
            self._results["sublists"][dataset] = []
            for subject_file in Path(self.inputs.config["features_dir"][dataset]).iterdir():
                if dataset in ["HCP-A", "HCP-D"]:
                    subject = subject_file.stem.split("_V1_MR")[0]
                elif dataset == "HCP-YA":
                    subject = subject_file.stem
                else:
                    raise DatasetError()
                check = self._check_sub(subject_file)
                if check:
                    self._results["sublists"][dataset].append(subject)

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
