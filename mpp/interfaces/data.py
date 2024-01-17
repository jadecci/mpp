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
    target = traits.Str(desc="target phenotype to predict")


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
        self._results["target"] = self.inputs.target

        return runtime


class _PredictionCombineInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    results = traits.List(dtype=dict, desc="accuracy results")
    features = traits.List(dtype=str, mandatory=True, desc="feature types")


class _PredictionCombineOutputSpec(TraitedSpec):
    results = traits.Dict(desc="accuracy results")


class PredictionCombine(SimpleInterface):
    """Combine prediction results across prediction targets"""
    input_spec = _PredictionCombineInputSpec
    output_spec = _PredictionCombineOutputSpec

    def _run_interface(self, runtime):
        for feature, results in zip(self.inputs.features, self.inputs.results):
            for key, val in results:
                self._results["results"][f"{key}_{feature}"] = val

        return runtime


class _PredictionSaveInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    results = traits.List(dtype=dict, desc="accuracy results")
    type = traits.Str(mandatory=True, desc="type of model used in prediction")
    target = traits.Str(mandatory=True, desc="prediction target")


class PredictionSave(SimpleInterface):
    """Save prediction results"""
    input_spec = _PredictionSaveInputSpec

    def _run_interface(self, runtime):
        output_file = Path(
            self.inputs.config["output_dir"], f"{self.inputs.type}_{self.inputs.target}.h5")
        results = {key: val for d in self.inputs.results for key, val in d.items()}
        pd.DataFrame(results).to_hdf(output_file, self.inputs.type)

        return runtime
