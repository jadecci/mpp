from pathlib import Path

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import numpy as np
import pandas as pd

from mpp.utilities.data import read_h5
from mpp.utilities.features import diffusion_mapping, score

base_dir = Path(__file__).resolve().parent.parent


class _CVFeaturesInputSpec(BaseInterfaceInputSpec):
    features_dir = traits.Dict(
        mandatory=True, dtype=str, desc='absolute path to extracted features for each dataset')
    sublists = traits.Dict(
        mandatory=True, dtype=list, desc='list of subjects available in each dataset')
    cv_split = traits.Dict(
        mandatory=True, dtype=list, desc='list of subjects in the test split of each fold')
    repeat = traits.Int(mandatory=True, desc='current repeat of cross-validation')
    fold = traits.Int(mandatory=True, desc='current fold in the repeat')
    level = traits.Str(mandatory=True, desc='parcellation level')
    config = traits.Dict({}, desc='configuration settings')


class _CVFeaturesOutputSpec(TraitedSpec):
    embeddings = traits.Dict(desc='embeddings for gradients')
    params = traits.Dict(desc='parameters for anatomical connectivity')
    repeat = traits.Int(desc='current repeat of cross-validation')
    fold = traits.Int(desc='current fold in the repeat')
    level = traits.Str(desc='parcellation level')


class CVFeatures(SimpleInterface):
    """Compute gradient embedding (diffusion mapping) and
    anatomical connectivity (structural co-registration)"""
    input_spec = _CVFeaturesInputSpec
    output_spec = _CVFeaturesOutputSpec

    def _extract_features(self) -> dict:
        all_sub = sum(self.inputs.sublists.values(), [])
        image_features = dict.fromkeys(all_sub)

        for subject in all_sub:
            dataset = [
                key for key in self.inputs.sublists if subject in self.inputs.sublists[key]][0]
            if dataset in ['HCP-A', 'HCP-D']:
                feature_file = Path(
                    self.inputs.features_dir[dataset], f'{dataset}_{subject}_V1_MR.h5')
            else:
                feature_file = Path(self.inputs.features_dir[dataset], f'{dataset}_{subject}.h5')

            feature_sub = {
                'rsfc': read_h5(feature_file, f'/rsfc/level{self.inputs.level}'),
                'myelin': read_h5(feature_file, f'/myelin/level{self.inputs.level}')}
            for stats in ['GMV', 'CS', 'CT']:
                ds_morph = f'/morphometry/{stats}/level{self.inputs.level}'
                feature_sub[stats] = read_h5(feature_file, ds_morph)
            image_features[subject] = feature_sub

        return image_features

    def _compute_features(self, image_features: dict, key: str) -> tuple[np.ndarray, pd.DataFrame]:
        train_sub = self.inputs.cv_split[key]
        embed = diffusion_mapping(image_features, train_sub, 'rsfc')
        for feature in ['GMV', 'CS', 'CT', 'myelin']:
            params = score(image_features, train_sub, feature)

        return embed, params

    def _run_interface(self, runtime):
        image_features = self._extract_features()
        self._results['embeddings'] = {}
        self._results['params'] = {}

        key = f'repeat{self.inputs.repeat}_fold{self.inputs.fold}'
        embed, params = self._compute_features(image_features, key)
        self._results['embeddings']['embedding'] = embed
        self._results['params']['params'] = params

        for inner in range(5):
            embed, params = self._compute_features(image_features, f'{key}_inner{inner}')
            self._results['embeddings'][f'embedding_inner{inner}'] = embed
            self._results['params'][f'params_inner{inner}'] = params

        self._results['repeat'] = self.inputs.repeat
        self._results['fold'] = self.inputs.fold
        self._results['level'] = self.inputs.level

        return runtime
