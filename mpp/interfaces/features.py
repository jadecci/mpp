from pathlib import Path

from mapalign.embed import compute_diffusion_map
from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
from sklearn.metrics import pairwise_distances
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd

from mpp.utilities import find_sub_file, fc_to_matrix


class _CVFeaturesInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    sublists = traits.Dict(
        mandatory=True, dtype=list, desc="list of subjects available in each dataset")
    cv_split = traits.Dict(mandatory=True, dtype=list, desc="test subjects of each fold")
    repeat = traits.Int(mandatory=True, desc="current repeat of cross-validation")
    fold = traits.Int(mandatory=True, desc="current fold in the repeat")
    target = traits.Str(mandatory=True, desc="prediction target")


class _CVFeaturesOutputSpec(TraitedSpec):
    cv_features_file = traits.File(exists=True, desc="file containing CV features")
    repeat = traits.Int(desc="current repeat of cross-validation")
    fold = traits.Int(desc="current fold in the repeat")


class CVFeatures(SimpleInterface):
    """Compute gradient embedding (diffusion mapping)
    and anatomical connectivity (structural co-registration)"""
    input_spec = _CVFeaturesInputSpec
    output_spec = _CVFeaturesOutputSpec

    def save_features(self, features: pd.DataFrame, key: str) -> None:
        features.to_hdf(self._results["cv_features_file"], key)

    def _compute_features(self, key: str, l_key: str, postfix: str = "") -> None:
        sublist = self.inputs.cv_split[key]
        nparc_dict = {"1": 116, "2": 232, "3": 350, "4": 454}

        # Diffusion mapping
        nparc = nparc_dict[self.inputs.config["level"]]
        rsfc = np.zeros((nparc, nparc, len(sublist)))
        for sub_i, subject in enumerate(sublist):
            subject_file, _, _ = find_sub_file(
                self.inputs.sublists, self.inputs.config["features_dir"], subject)
            rsfc_sub = pd.read_hdf(subject_file, f"rs_sfc_{l_key}")
            rsfc[:, :, sub_i] = fc_to_matrix(rsfc_sub, nparc)
        rsfc_thresh = np.tanh(rsfc.mean(axis=2))
        for i in range(rsfc_thresh.shape[0]):
            rsfc_thresh[i, rsfc_thresh[i, :] < np.percentile(rsfc_thresh[i, :], 90)] = 0
        rsfc_thresh[rsfc_thresh < 0] = 0  # there should be very few negatives
        affinity = 1 - pairwise_distances(rsfc_thresh, metric='cosine')
        embed = compute_diffusion_map(affinity, alpha=0.5) # Nparc x Ngrad -> Ngrad x Nparc
        pd.DataFrame(embed.T).to_hdf(self._results["cv_features_file"], f"embed{postfix}")

        # Structural Co-Registration
        # see https://github.com/katielavigne/score/blob/main/score.py
        for feature in ["gmv", "cs", "ct"]:
            if feature == "gmv":
                nparc = nparc_dict[self.inputs.config["level"]]
            else:
                nparc = int(self.inputs.config["level"]) * 100
            features = pd.DataFrame()
            params = pd.DataFrame()
            for sub_i, subject in enumerate(sublist):
                subject_file, _, _ = find_sub_file(
                    self.inputs.sublists, self.inputs.config["features_dir"], subject)
                features_sub = pd.read_hdf(subject_file, f"s_{feature}_{l_key}")
                features = pd.concat([features, features_sub], axis="index")
            features[features.columns] = features[features.columns].apply(pd.to_numeric)
            features.columns = range(nparc)
            features = features.join(pd.DataFrame({"mean": features.mean(axis=1)}))
            for i in range(nparc):
                for j in range(nparc):
                    results = ols(f"features[{i}] ~ features[{j}] + mean", data=features).fit()
                    params[f"{i}_{j}"] = [
                        results.params["Intercept"], results.params[f"features[{j}]"],
                        results.params["mean"]]
            params.to_hdf(self._results["cv_features_file"], f"params_{feature}{postfix}")

    def _run_interface(self, runtime):
        tmp_dir = Path(self.inputs.config["work_dir"], f"{self.inputs.target}_features_tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        key = f"repeat{self.inputs.repeat}_fold{self.inputs.fold}"
        l_key = f"level{self.inputs.config['level']}"
        self._results["cv_features_file"] = Path(tmp_dir, f"cv_features_{key}_{l_key}.h5")

        self._compute_features(key, l_key)
        for inner in range(5):
            self._compute_features(f"{key}_inner{inner}", l_key, postfix=f"_inner{inner}")
        self._results["repeat"] = self.inputs.repeat
        self._results["fold"] = self.inputs.fold

        return runtime
