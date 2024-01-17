from pathlib import Path

from mapalign.embed import compute_diffusion_map
from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
from sklearn.metrics import pairwise_distances
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd

from mpp.utilities import find_sub_file


class _CVFeaturesInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    sublists = traits.Dict(
        mandatory=True, dtype=list, desc="list of subjects available in each dataset")
    cv_split = traits.Dict(mandatory=True, dtype=list, desc="test subjects of each fold")
    repeat = traits.Int(mandatory=True, desc="current repeat of cross-validation")
    fold = traits.Int(mandatory=True, desc="current fold in the repeat")
    level = traits.Str(mandatory=True, desc="parcellation level")


class _CVFeaturesOutputSpec(TraitedSpec):
    embeddings = traits.Dict(desc="embeddings for gradients")
    params_gmv = traits.Dict(desc="parameters for GMV-based anatomical connectivity")
    params_ct = traits.Dict(desc="parameters for CT-based anatomical connectivity")
    params_cs = traits.Dict(desc="parameters for CS-based anatomical connectivity")
    repeat = traits.Int(desc="current repeat of cross-validation")
    fold = traits.Int(desc="current fold in the repeat")
    level = traits.Str(desc="parcellation level")


class CVFeatures(SimpleInterface):
    """Compute gradient embedding (diffusion mapping)
    and anatomical connectivity (structural co-registration)"""
    input_spec = _CVFeaturesInputSpec
    output_spec = _CVFeaturesOutputSpec

    @staticmethod
    def _rsfc_to_matrix(data_in: pd.DataFrame, nparc: int) -> np.ndarray:
        arr_out = np.zeros((nparc, nparc))
        for i in range(nparc):
            for j in range(i+1, nparc):
                arr_out[i, j] = float(data_in[f"rs_sfc_{i}_{j}"].values[0])
                arr_out[j, i] = arr_out[i, j]
        return arr_out

    def _compute_features(self, sublist: list) -> tuple[np.ndarray, dict]:
        nparc_dict = {"1": 116, "2": 232, "3": 350, "4": 454}
        nparc = nparc_dict[self.inputs.level]
        l_key = f"level{self.inputs.config['level']}"

        # Diffusion mapping
        rsfc = np.zeros((nparc, nparc, len(sublist)))
        for sub_i, subject in enumerate(sublist):
            subject_file = find_sub_file(
                self.inputs.sublists, self.inputs.config["features_dir"], subject)
            rsfc_sub = pd.DataFrame(pd.read_hdf(subject_file, f"rs_sfc_{l_key}"))
            rsfc[:, :, sub_i] = self._rsfc_to_matrix(rsfc_sub, nparc)
        rsfc_thresh = np.tanh(rsfc.mean(axis=2))
        for i in range(rsfc_thresh.shape[0]):
            rsfc_thresh[i, rsfc_thresh[i, :] < np.percentile(rsfc_thresh[i, :], 90)] = 0
        rsfc_thresh[rsfc_thresh < 0] = 0  # there should be very few negatives
        affinity = 1 - pairwise_distances(rsfc_thresh, metric='cosine')
        embed = compute_diffusion_map(affinity, alpha=0.5)

        # Structural Co-Registration
        # see https://github.com/katielavigne/score/blob/main/score.py
        params = {"gmv": pd.DataFrame(), "cs": pd.DataFrame(), "ct": pd.DataFrame()}
        for feature in ["gmv", "cs", "ct"]:
            features = pd.DataFrame()
            for sub_i, subject in enumerate(sublist):
                subject_file = find_sub_file(
                    self.inputs.sublists, self.inputs.config["features_dir"], subject)
                features_sub = pd.DataFrame(pd.read_hdf(subject_file, f"s_{feature}_{l_key}"))
                features = pd.concat([features, features_sub], axis="index")
            features = features.join(pd.DataFrame({"mean": features.mean(axis=1)}))
            features[features.columns] = features[features.columns].apply(pd.to_numeric)
            features.columns = range(nparc)
            for i in range(nparc):
                for j in range(nparc):
                    results = ols(f"features[{i}] ~ features[{j}] + mean", data=features).fit()
                    params[feature][f"{i}_{j}"] = [
                        results.params["Intercept"], results.params[f"features[{j}]"],
                        results.params["mean"]]

        return embed, params

    def _run_interface(self, runtime):
        key = f"repeat{self.inputs.repeat}_fold{self.inputs.fold}"
        embed, params = self._compute_features(self.inputs.cv_split[key])
        self._results["embeds"]["embed"] = embed
        self._results["params_gmv"]["params"] = params["gmv"]
        self._results["params_ct"]["params"] = params["ct"]
        self._results["params_cs"]["params"] = params["cs"]
        for inner in range(5):
            embed, params = self._compute_features(self.inputs.cv_split[f"{key}_inner{inner}"])
            self._results["embeds"][f"embed_inner{inner}"] = embed
            self._results["params_gmv"][f"params_inner{inner}"] = params["gmv"]
            self._results["params_ct"][f"params_inner{inner}"] = params["ct"]
            self._results["params_cs"][f"params_inner{inner}"] = params["cs"]
        self._results["repeat"] = self.inputs.repeat
        self._results["fold"] = self.inputs.fold
        self._results["level"] = self.inputs.level

        return runtime
