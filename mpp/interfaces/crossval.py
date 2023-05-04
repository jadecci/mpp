import itertools
from pathlib import Path

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.stats.multitest import multipletests

from mpp.utilities.models import elastic_net, permutation_test
from mpp.utilities.data import write_h5, cv_extract_data


class _CrossValSplitInputSpec(BaseInterfaceInputSpec):
    sublists = traits.Dict(mandatory=True, dtype=list, desc='availabel subjects in each dataset')
    config = traits.Dict(mandatory=True, desc='configuration settings')
    permutation = traits.Bool(False, desc='use n_repeats_perm instead')


class _CrossValSplitOutputSpec(TraitedSpec):
    cv_split = traits.Dict(dtype=list, desc='list of subjects in the test split of each fold')


class CrossValSplit(SimpleInterface):
    """Generate cross-validation splits"""
    input_spec = _CrossValSplitInputSpec
    output_spec = _CrossValSplitOutputSpec

    def _run_interface(self, runtime):
        subjects = sum(self.inputs.sublists.values(), [])
        datasets = [[ds] * len(self.inputs.sublists[ds]) for ds in self.inputs.sublists]
        datasets = list(itertools.chain.from_iterable(datasets))

        self._results['cv_split'] = {}
        if self.inputs.permutation:
            n_repeats = int(self.inputs.config['n_repeats_perm'])
        else:
            n_repeats = int(self.inputs.config['n_repeats'])
        n_folds = int(self.inputs.config['n_folds'])
        rskf = RepeatedStratifiedKFold(
            n_splits=n_folds, n_repeats=n_repeats, random_state=int(self.inputs.config['cv_seed']))
        for fold, (_, test_ind) in enumerate(rskf.split(subjects, datasets)):
            key = f'repeat{int(np.floor(fold/n_folds))}_fold{int(fold%n_folds)}'
            self._results['cv_split'][key] = np.array(subjects)[test_ind]

        return runtime


class _RegionwiseModelInputSpec(BaseInterfaceInputSpec):
    sublists = traits.Dict(mandatory=True, dtype=list, desc='available subjects in each dataset')
    features_dir = traits.Dict(
        mandatory=True, dtype=str, desc='absolute path to extracted features for each dataset')
    phenotypes = traits.Dict(mandatory=True, dtype=float, desc='phenotype values')
    phenotypes_perm = traits.Dict({}, dtype=float, desc='shuffled phenotype values for permutation')

    embeddings = traits.Dict(mandatory=True, desc='embeddings for gradients')
    params = traits.Dict(mandatory=True, desc='parameters for anatomical connectivity')

    cv_split = traits.Dict(mandatory=True, dtype=list, desc='test subjects of each fold')
    cv_split_perm = traits.Dict({}, dtype=list, desc='test subjects of each fold')

    level = traits.Str(mandatory=True, desc='parcellation level (1 to 4)')
    repeat = traits.Int(mandatory=True, desc='current repeat of cross-validation')
    fold = traits.Int(mandatory=True, desc='current fold in the repeat')

    mode = traits.Str(
        mandatory=True, desc=("'validate' to train and validate models with permutation tests, "
                              "'test' to train and test models on test folds"))
    selected = traits.Dict(dtype=list, desc='selected regions for each parcellation level')
    config = traits.Dict(mandatory=True, desc='configuration settings')


class _RegionwiseModelOutputSpec(TraitedSpec):
    results = traits.Dict(desc='accuracy results')
    selected = traits.Dict(desc='whether each feature is selected')


class RegionwiseModel(SimpleInterface):
    """Train and validate, or test, a region-wise model"""
    input_spec = _RegionwiseModelInputSpec
    output_spec = _RegionwiseModelOutputSpec

    def _validate(self, repeat: int, permutation: bool = False) -> tuple[np.ndarray, ...]:
        if permutation:
            cv_split = self.inputs.cv_split_perm
            phenotypes = self.inputs.phenotypes_perm
        else:
            cv_split = self.inputs.cv_split
            phenotypes = self.inputs.phenotypes
        n_folds = int(self.inputs.config["n_folds"])
        all_sub = sum(self.inputs.sublists.values(), [])

        test_sub = cv_split[f'repeat{repeat}_fold{self.inputs.fold}']
        val_sub = cv_split[f'repeat{repeat}_fold{(self.inputs.fold+1)%n_folds}']
        testval_sub = np.concatenate((val_sub, test_sub))
        train_sub = [subject for subject in all_sub if subject not in testval_sub]

        train_x, train_y = cv_extract_data(
            self.inputs.sublists, self.inputs.features_dir, train_sub, repeat, self.inputs.level,
            self.inputs.embeddings, self.inputs.params, phenotypes, permutation=permutation)
        val_x, val_y = cv_extract_data(
            self.inputs.sublists, self.inputs.features_dir, val_sub, repeat, self.inputs.level,
            self.inputs.embeddings, self.inputs.params, phenotypes, permutation=permutation)

        r = np.zeros(train_x.shape[2])
        cod = np.zeros(train_x.shape[2])
        for region in range(train_x.shape[2]):
            r[region], cod[region], _, _ = elastic_net(
                train_x[:, :, region], train_y, val_x[:, :, region], val_y,
                int(self.inputs.config['n_alphas']))
        r = np.nan_to_num(r)

        return r, cod

    def _test(self) -> tuple[np.ndarray, ...]:
        all_sub = sum(self.inputs.sublists.values(), [])
        test_sub = self.inputs.cv_split[f'repeat{self.inputs.repeat}_fold{self.inputs.fold}']
        train_sub = [subject for subject in all_sub if subject not in test_sub]

        train_x, train_y = cv_extract_data(
            self.inputs.sublists, self.inputs.features_dir, train_sub, self.inputs.repeat,
            self.inputs.level, self.inputs.embeddings, self.inputs.params, self.inputs.phenotypes)
        test_x, test_y = cv_extract_data(
            self.inputs.sublists, self.inputs.features_dir, test_sub, self.inputs.repeat,
            self.inputs.level, self.inputs.embeddings, self.inputs.params, self.inputs.phenotypes)

        r = np.zeros(train_x.shape[2])
        cod = np.zeros(train_x.shape[2])
        coef = np.zeros((train_x.shape[2], train_x.shape[1]))
        l1_ratios = np.zeros(train_x.shape[2])
        for region in range(train_x.shape[2]):
            if self.inputs.selected[f'regions_level{self.inputs.level}'][region]:
                r[region], cod[region], coef[region, :], l1_ratios[region] = elastic_net(
                    train_x[:, :, region], train_y, test_x[:, :, region], test_y,
                    int(self.inputs.config['n_alphas']))

        return r, cod, coef, l1_ratios

    def _run_interface(self, runtime):
        self._results['results'] = {}

        if self.inputs.mode == 'validate':
            r, cod = self._validate(self.inputs.repeat)
            key = f'repeat{self.inputs.repeat}_fold{self.inputs.fold}_level{self.inputs.level}'
            self._results['results'][f'r_{key}'] = r
            self._results['results'][f'cod_{key}'] = cod

            # assuming n_repeats_perm = n_repeats x 10
            n_perm_check = int(self.inputs.config['n_repeats']) * 10
            if int(self.inputs.config['n_repeats_perm']) == n_perm_check:
                for repeat_split in range(10):
                    repeat = int(self.inputs.repeat) * 10 + repeat_split
                    key = f'repeat{repeat}_fold{self.inputs.fold}_level{self.inputs.level}'
                    r, cod = self._validate(repeat, permutation=True)
                    self._results['results'][f'r_perm_{key}'] = r
                    self._results['results'][f'cod_perm_{key}'] = cod
            else:
                raise ValueError("'n_repeats_perm' must be 10 x 'n_repeats'")

        elif self.inputs.mode == 'test':
            key = f'repeat{self.inputs.repeat}_fold{self.inputs.fold}_level{self.inputs.level}'
            r, cod, coef, l1_ratios = self._test()
            self._results['results'][f'r_{key}'] = r
            self._results['results'][f'cod_{key}'] = cod
            self._results['results'][f'l1ratio_{key}'] = l1_ratios
            self._results['selected'] = {}
            self._results['selected'][f'features_{key}'] = list(coef != 0)

        return runtime


class _RegionSelectInputSpec(BaseInterfaceInputSpec):
    results = traits.List(dtype=dict, desc='accuracy results')
    levels = traits.List(mandatory=True, desc='parcellation levels (1 to 4)')
    output_dir = traits.Directory(mandatory=True, desc='absolute path to output directory')
    overwrite = traits.Bool(mandatory=True, desc='whether to overwrite existing results')
    config = traits.Dict(mandatory=True, desc='configuration settings')


class _RegionSelectOutputSpec(TraitedSpec):
    selected = traits.Dict(dtype=list, desc='selected regions for each parcellation level')


class RegionSelect(SimpleInterface):
    """Select region-wise models that pass permutation tests"""
    input_spec = _RegionSelectInputSpec
    output_spec = _RegionSelectOutputSpec

    def _extract_results(self, results_dict: dict,  permutation: bool = False) -> np.ndarray:
        if permutation:
            n_repeats = self.inputs.config['n_repeats_perm']
            key = f'{self.inputs.config["features_criteria"]}_perm'
        else:
            n_repeats = self.inputs.config['n_repeats']
            key = self.inputs.config['features_criteria']
        n_folds = self.inputs.config['n_folds']

        n_region_dict = {'1': 116, '2': 232, '3': 350, '4': 454}
        n_regions = [n_region_dict[level] for level in self.inputs.levels]

        results = np.zeros((int(n_repeats), sum(n_regions)))
        for repeat in range(int(n_repeats)):
            for fold in range(int(n_folds)):
                pos = 0
                for level in self.inputs.levels:
                    key_curr = f'{key}_repeat{repeat}_fold{fold}_level{level}'
                    regions = range(pos, pos + n_region_dict[level])
                    pos = pos + n_region_dict[level]
                    results[repeat, regions] = results[repeat, regions] + results_dict[key_curr]
            results[repeat, :] = np.divide(results[repeat, :], int(n_folds))

        return results

    def _save_results(self) -> None:
        Path(self.inputs.output_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(self.inputs.output_dir, 'regionwise_results.h5')
        for key, val in self._results['selected'].items():
            write_h5(output_file, f'/{key}', np.array(val), self.inputs.overwrite)

    def _run_interface(self, runtime):
        n_region_dict = {'1': 116, '2': 232, '3': 350, '4': 454}

        results_dict = {key: item for d in self.inputs.results for key, item in d.items()}
        results = self._extract_results(results_dict)
        results_perm = self._extract_results(results_dict, permutation=True)

        p = permutation_test(results.mean(axis=0), results_perm)
        selected = multipletests(p, method='fdr_bh')[0]

        self._results['selected'] = {}
        pos = 0
        for level in self.inputs.levels:
            regions = range(pos, pos + n_region_dict[level])
            pos = pos + n_region_dict[level]
            self._results['selected'][f'regions_level{level}'] = selected[regions]

        self._save_results()

        return runtime


class _IntegratedFeaturesModelInputSpec(BaseInterfaceInputSpec):
    sublists = traits.Dict(mandatory=True, dtype=list, desc='available subjects in each dataset')
    features_dir = traits.Dict(
        mandatory=True, dtype=str, desc='absolute path to extracted features for each dataset')
    phenotypes = traits.Dict(mandatory=True, dtype=float, desc='phenotype values')
    embeddings = traits.Dict(mandatory=True, desc='embeddings for gradients')
    params = traits.Dict(mandatory=True, desc='parameters for anatomical connectivity')
    cv_split = traits.Dict(mandatory=True, dtype=list, desc='test subjects of each fold')

    level = traits.Str(mandatory=True, desc='parcellation level (1 to 4)')
    repeat = traits.Int(mandatory=True, desc='current repeat of cross-validation')
    fold = traits.Int(mandatory=True, desc='current fold in the repeat')

    selected_features = traits.Dict(mandatory=True, desc='whether each feature is selected')
    config = traits.Dict(mandatory=True, desc='configuration settings')


class _IntegratedFeaturesModelOutputSpec(TraitedSpec):
    results = traits.Dict(desc='accuracy results')


class IntegratedFeaturesModel(SimpleInterface):
    """Train and test the integrated features model"""
    input_spec = _IntegratedFeaturesModelInputSpec
    output_spec = _IntegratedFeaturesModelOutputSpec

    def _run_interface(self, runtime):
        all_sub = sum(self.inputs.sublists.values(), [])
        test_sub = self.inputs.cv_split[f'repeat{self.inputs.repeat}_fold{self.inputs.fold}']
        train_sub = [subject for subject in all_sub if subject not in test_sub]
        selected_features = np.array(self.inputs.selected_features[
            f'features_repeat{self.inputs.repeat}_fold{self.inputs.fold}_level{self.inputs.level}'])

        train_x, train_y = cv_extract_data(
            self.inputs.sublists, self.inputs.features_dir, train_sub, self.inputs.repeat,
            self.inputs.level, self.inputs.embeddings, self.inputs.params, self.inputs.phenotypes,
            selected_features=selected_features)
        test_x, test_y = cv_extract_data(
            self.inputs.sublists, self.inputs.features_dir, test_sub, self.inputs.repeat,
            self.inputs.level, self.inputs.embeddings, self.inputs.params, self.inputs.phenotypes,
            selected_features=selected_features)

        r, cod, coef, l1_ratio = elastic_net(
            train_x, train_y, test_x, test_y, int(self.inputs.config['n_alphas']))

        key = f'repeat{self.inputs.repeat}_fold{self.inputs.fold}_level{self.inputs.level}'
        self._results['results'] = {
            f'r_{key}': r, f'cod_{key}': cod, f'l1ratio_{key}': l1_ratio,
            f'features_{key}': list(coef != 0)}

        return runtime
