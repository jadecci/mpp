from pathlib import Path

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import numpy as np
import pandas as pd

from mpp.exceptions import DatasetError
from mpp.utilities.data import dataset_params, write_h5


class _InitFeaturesInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")


class _InitFeaturesOutputSpec(TraitedSpec):
    sublists = traits.Dict(dtype=list, desc="list of subjects available in each dataset")


class InitFeatures(SimpleInterface):
    """Extract available phenotype data and required list of subjects"""
    input_spec = _InitFeaturesInputSpec
    output_spec = _InitFeaturesOutputSpec

    def _check_pheno_conf(self, subject_file: Path):
        return

    def _run_interface(self, runtime):
        self._results["sublists"] = {}

        for dataset in Path(self.inputs.config["features_dir"]).iterdir():
            if not dataset.name.startswith("."):
                sublist = []
                for subject_file in dataset.iterdir():
                    if dataset in ["HCP-A", "HCP-D"]:
                        subject = subject_file.stem.split("_V1_MR")[0]
                    elif dataset == "HCP-YA":
                        subject = subject_file.stem
                    else:
                        raise DatasetError()
                    sublist.append(subject)
                self._results["sublists"][dataset] = sublist

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
