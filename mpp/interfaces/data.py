from pathlib import Path

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import pandas as pd

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
        if pheno[self.inputs.target].isnull().values[0] or pheno[self.inputs.target] == 999:
            return False
        if conf.isnull().values.any():
            return False
        return True

    def _run_interface(self, runtime):
        self._results["sublists"] = {}
        for dataset, listf in zip(self.inputs.config["datasets"], self.inputs.config["sublists"]):
            sublist = pd.read_table(listf).squeeze("columns")
            self._results["sublists"][dataset] = []
            for subject in sublist:
                subject_file = Path(self.inputs.config["features_dir"], dataset, f"{subject}.h5")
                if dataset in ["HCP-A", "HCP-D"]:
                    subject_id = subject.split("_V1_MR")[0]
                elif dataset == "HCP-YA":
                    subject_id = subject
                else:
                    raise DatasetError()
                if self._check_sub(subject_file):
                    self._results["sublists"][dataset].append(subject_id)
        self._results["target"] = self.inputs.target

        return runtime


class _PredictionCombineInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    results = traits.List(dtype=dict, desc="accuracy results")


class _PredictionCombineOutputSpec(TraitedSpec):
    results = traits.Dict(desc="accuracy results")


class PredictionCombine(SimpleInterface):
    """Combine prediction results across features"""
    input_spec = _PredictionCombineInputSpec
    output_spec = _PredictionCombineOutputSpec

    def _run_interface(self, runtime):
        self._results["results"] = {}
        for results in self.inputs.results:
            for key, val in results.items():
                self._results["results"].update({key: val})

        return runtime


class _PredictionSaveInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    results = traits.List(dtype=dict, desc="accuracy results")
    model_type = traits.Str(mandatory=True, desc="type of model used in prediction")
    target = traits.Str(mandatory=True, desc="prediction target")


class PredictionSave(SimpleInterface):
    """Save prediction results"""
    input_spec = _PredictionSaveInputSpec

    def _run_interface(self, runtime):
        output_file = Path(
            self.inputs.config["output_dir"], f"{self.inputs.model_type}_{self.inputs.target}.h5")
        for results in self.inputs.results:
            for key, val in results.items():
                if val.ndim == 0:
                    pd.DataFrame({key: val}, index=[0]).to_hdf(output_file, key)
                else:
                    pd.DataFrame({key: val}).to_hdf(output_file, key)

        return runtime
