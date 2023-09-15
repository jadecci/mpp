import itertools
from pathlib import Path

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.stats.multitest import multipletests

from mpp.utilities.models import (
    elastic_net, kernel_ridge_corr_cv, permutation_test)
from mpp.utilities.data import write_h5, cv_extract_data, cv_extract_subject_data

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
    rw_ypred = traits.Dict(desc='Predicted psychometric values')


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
            r[region], cod[region], _ = elastic_net(
                train_x[:, :, region], train_y, val_x[:, :, region], val_y,
                int(self.inputs.config['n_alphas']))
        r = np.nan_to_num(r)

        return r, cod

    def _test(self) -> None:
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
        coef = np.zeros((train_x.shape[2], train_x.shape[1]+1))
        train_ypred = np.zeros((len(train_y), train_x.shape[2]))
        test_ypred = np.zeros((len(test_y), train_x.shape[2]))
        for region in range(train_x.shape[2]):
            # if self.inputs.selected[f'regions_level{self.inputs.level}'][region]:
            r[region], cod[region], model = elastic_net(
                train_x[:, :, region], train_y, test_x[:, :, region], test_y,
                int(self.inputs.config['n_alphas']))
            coef[region, :] = np.concatenate((model.coef_, [model.intercept_]))
            train_ypred[:, region] = model.predict(train_x[:, :, region])
            test_ypred[:, region] = model.predict(test_x[:, :, region])

        key = f'repeat{self.inputs.repeat}_fold{self.inputs.fold}_level{self.inputs.level}'
        self._results['results'][f'r_{key}'] = r
        self._results['results'][f'cod_{key}'] = cod
        self._results['results'][f'model_{key}'] = coef
        self._results['rw_ypred'] = {'train_ypred': train_ypred, 'test_ypred': test_ypred}

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
            self._test()

        return runtime


class _RegionSelectInputSpec(BaseInterfaceInputSpec):
    results = traits.List(mandatory=True, dtype=dict, desc='accuracy results')
    levels = traits.List(mandatory=True, desc='parcellation levels (1 to 4)')
    output_dir = traits.Directory(mandatory=True, desc='absolute path to output directory')
    overwrite = traits.Bool(mandatory=True, desc='whether to overwrite existing results')
    config = traits.Dict(mandatory=True, desc='configuration settings')
    phenotype = traits.Str(mandatory=True, desc='phenotype to use as prediction target')


class _RegionSelectOutputSpec(TraitedSpec):
    selected = traits.Dict(dtype=list, desc='selected regions for each parcellation level')


class RegionSelect(SimpleInterface):
    """Select region-wise models that pass permutation tests"""
    input_spec = _RegionSelectInputSpec
    output_spec = _RegionSelectOutputSpec

    def _extract_results(
            self, results_dict: dict, level: str,  permutation: bool = False) -> np.ndarray:
        if permutation:
            n_repeats = self.inputs.config['n_repeats_perm']
            key = f'{self.inputs.config["features_criteria"]}_perm'
        else:
            n_repeats = self.inputs.config['n_repeats']
            key = self.inputs.config['features_criteria']
        n_folds = self.inputs.config['n_folds']

        n_region_dict = {'1': 116, '2': 232, '3': 350, '4': 454}

        results = np.zeros((int(n_repeats), n_region_dict[level]))
        for repeat in range(int(n_repeats)):
            for fold in range(int(n_folds)):
                key_curr = f'{key}_repeat{repeat}_fold{fold}_level{level}'
                results[repeat, :] = results[repeat, :] + results_dict[key_curr]
            results[repeat, :] = np.divide(results[repeat, :], int(n_folds))

        return results

    def _save_results(self) -> None:
        Path(self.inputs.output_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(self.inputs.output_dir, f'regionwise_results_{self.inputs.phenotype}.h5')
        for key, val in self._results['selected'].items():
            write_h5(output_file, f'/{key}', np.array(val), self.inputs.overwrite)

    def _run_interface(self, runtime):
        results_dict = {key: item for d in self.inputs.results for key, item in d.items()}

        self._results['selected'] = {}
        for level in self.inputs.levels:
            results = self._extract_results(results_dict, level)
            results_perm = self._extract_results(results_dict, level, permutation=True)

            p = permutation_test(results.mean(axis=0), results_perm)
            selected = multipletests(p, method='fdr_bh')[0]
            self._results['selected'][f'regions_level{level}'] = selected
        self._save_results()

        return runtime


class _ModalitywiseModelInputSpec(BaseInterfaceInputSpec):
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
    config = traits.Dict(mandatory=True, desc='configuration settings')


class _ModalitywiseModelOutputSpec(TraitedSpec):
    results = traits.Dict(desc='accuracy results')
    mw_ypred = traits.Dict(desc='Predicted psychometric values')


class ModalitywiseModel(SimpleInterface):
    """Train and test modality-wise models"""
    input_spec = _ModalitywiseModelInputSpec
    output_spec = _ModalitywiseModelOutputSpec

    def _extract_data(self, subjects: list) -> tuple[dict, np.ndarray,]:
        y = np.zeros(len(subjects))
        x_mod = {}

        for i, subject in enumerate(subjects):
            x = cv_extract_subject_data(
                self.inputs.sublists, subject, self.inputs.features_dir, self.inputs.level,
                False, self.inputs.embeddings, self.inputs.params,
                self.inputs.repeat)
            x_rsfmri = np.concatenate((
                x[0][np.triu_indices_from(x[0], k=1)],
                x[1][np.triu_indices_from(x[1], k=1)],
                x[2][np.triu_indices_from(x[2], k=1)],
                x[4], x[5], x[6], x[7], x[12].flatten()))
            x_tfmri = np.concatenate(([
                x[3][:, :, i][np.triu_indices_from(x[3][:, :, i], k=1)]
                for i in range(x[3].shape[2])]))
            x_smri = np.concatenate((
                x[8], x[9], x[10], x[11], x[13][np.triu_indices_from(x[13], k=1)],
                x[14][np.triu_indices_from(x[14], k=1)], x[15][np.triu_indices_from(x[15], k=1)]))
            if i == 0:
                x_mod = {
                    'x_rsfmri': x_rsfmri, 'x_tfmri': x_tfmri, 'x_smri': x_smri}
            else:
                x_mod = {
                    'x_rsfmri': np.vstack((x_mod['x_rsfmri'], x_rsfmri)),
                    'x_tfmri': np.vstack((x_mod['x_tfmri'], x_tfmri)),
                    'x_smri': np.vstack((x_mod['x_smri'], x_smri))}
            # TODO: diffusion features
            y[i] = self.inputs.phenotypes[subjects[i]]

        return x_mod, y

    def _test(self, train_x, train_y, test_x, test_y, modality):
        key = (f'{modality}_repeat{self.inputs.repeat}_fold{self.inputs.fold}'
               f'_level{self.inputs.level}')

        #r, cod, model = elastic_net(
        #    train_x, train_y, test_x, test_y, int(self.inputs.config['n_alphas']))
        #self._results['results'][f'en_r_{key}'] = r
        #self._results['results'][f'en_cod_{key}'] = cod

        r, cod, train_ypred, test_ypred = kernel_ridge_corr_cv(train_x, train_y, test_x, test_y)
        self._results['results'][f'krcorr_r_{key}'] = r
        self._results['results'][f'krcorr_cod_{key}'] = cod

        return train_ypred, test_ypred

    def _run_interface(self, runtime):
        self._results['results'] = {}
        all_sub = sum(self.inputs.sublists.values(), [])
        test_sub = self.inputs.cv_split[f'repeat{self.inputs.repeat}_fold{self.inputs.fold}']
        train_sub = [subject for subject in all_sub if subject not in test_sub]

        train_x, train_y = self._extract_data(train_sub)
        test_x, test_y = self._extract_data(test_sub)

        ypred = {}
        modality_list = ['rsfmri', 'tfmri', 'smri']
        iters = zip(train_x.values(), test_x.values(), modality_list)
        for train_curr, test_curr, modality in iters:
            train_pred, test_pred = self._test(train_curr, train_y, test_curr, test_y, modality)
            if modality == 'rsfmri_stat':
                ypred = {'train_ypred': train_pred, 'test_ypred': test_pred}
            else:
                ypred = {
                    'train_ypred': np.vstack((ypred['train_ypred'], train_pred)),
                    'test_ypred': np.vstack((ypred['test_ypred'], test_pred))}
        self._results['mw_ypred'] = {
            'train_ypred': ypred['train_ypred'].T, 'test_ypred': ypred['test_ypred'].T}

        return runtime


class _FeaturewiseModelInputSpec(BaseInterfaceInputSpec):
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
    config = traits.Dict(mandatory=True, desc='configuration settings')


class _FeaturewiseModelOutputSpec(TraitedSpec):
    results = traits.Dict(desc='accuracy results')
    fw_ypred = traits.Dict(desc='Predicted psychometric values')


class FeaturewiseModel(SimpleInterface):
    """Train and test feature-wise models"""
    input_spec = _FeaturewiseModelInputSpec
    output_spec = _FeaturewiseModelOutputSpec
    _feature_list = [
        'rsfc', 'dfc', 'efc', 'tfc', 'strength', 'betweenness', 'participation', 'efficiency',
        'myelin', 'gmv', 'cs', 'ct', 'gradients', 'ac_gmv', 'ac_cs', 'ac_ct']

    @staticmethod
    def _add_sub_data(data_dict, sub_data, key, ind):
        if ind == 0:
            data_dict[key] = sub_data
        else:
            data_dict[key] = np.vstack((data_dict[key], sub_data))
        return data_dict

    def _extract_data(self, subjects: list) -> tuple[dict, np.ndarray]:
        y = np.zeros(len(subjects))
        x_all = {}

        for i, subject in enumerate(subjects):
            x = cv_extract_subject_data(
                self.inputs.sublists, subject, self.inputs.features_dir, self.inputs.level,
                False, self.inputs.embeddings, self.inputs.params,
                self.inputs.repeat)
            for key, x_curr in zip(self._feature_list, x):
                if key in ['rsfc', 'dfc', 'efc', 'ac_gmv', 'ac_cs', 'ac_ct']:
                    x_feature = x_curr[np.triu_indices_from(x_curr, k=1)]
                    self._add_sub_data(x_all, x_feature, key, i)
                elif key == 'gradients':
                    self._add_sub_data(x_all, x_curr.flatten(), key, i)
                elif key == 'tfc':
                    for t_run in range(x_curr.shape[2]):
                        x_feature = x_curr[:, :, t_run]
                        x_feature = x_feature[np.triu_indices_from(x_feature, k=1)]
                        self._add_sub_data(x_all, x_feature, f'{key}_{t_run}', i)
                else:
                    self._add_sub_data(x_all, x_curr, key, i)
            # TODO: diffusion features
            y[i] = self.inputs.phenotypes[subjects[i]]
            if i == len(subjects)-1:
                for t_run in range(1, x[3].shape[2]):
                    self._feature_list.insert(3+t_run, 'tfc_{t_run}')

        return x_all, y

    def _test(self, train_x, train_y, test_x, test_y, feature):
        key = (f'{feature}_repeat{self.inputs.repeat}_fold{self.inputs.fold}'
               f'_level{self.inputs.level}')

        r, cod, _ = elastic_net(
            train_x, train_y, test_x, test_y, int(self.inputs.config['n_alphas']))
        self._results['results'][f'en_r_{key}'] = r
        self._results['results'][f'en_cod_{key}'] = cod

        r, cod, train_ypred, test_ypred = kernel_ridge_corr_cv(train_x, train_y, test_x, test_y)
        self._results['results'][f'krcorr_r_{key}'] = r
        self._results['results'][f'krcorr_cod_{key}'] = cod

        return train_ypred, test_ypred

    def _run_interface(self, runtime):
        self._results['results'] = {}
        all_sub = sum(self.inputs.sublists.values(), [])
        test_sub = self.inputs.cv_split[f'repeat{self.inputs.repeat}_fold{self.inputs.fold}']
        train_sub = [subject for subject in all_sub if subject not in test_sub]

        train_x, train_y = self._extract_data(train_sub)
        test_x, test_y = self._extract_data(test_sub)

        ypred = {}
        iters = zip(train_x.values(), test_x.values(), self._feature_list)
        for train_curr, test_curr, feature in iters:
            train_pred, test_pred = self._test(train_curr, train_y, test_curr, test_y, feature)
            if feature == 'rsfc':
                ypred = {'train_ypred': train_pred, 'test_ypred': test_pred}
            else:
                ypred = {
                    'train_ypred': np.vstack((ypred['train_ypred'], train_pred)),
                    'test_ypred': np.vstack((ypred['test_ypred'], test_pred))}
        self._results['fw_ypred'] = {
            'train_ypred': ypred['train_ypred'].T, 'test_ypred': ypred['test_ypred'].T}

        return runtime


class _IntegratedFeaturesModelInputSpec(BaseInterfaceInputSpec):
    sublists = traits.Dict(mandatory=True, dtype=list, desc='available subjects in each dataset')
    # features_dir = traits.Dict(
    #    mandatory=True, dtype=str, desc='absolute path to extracted features for each dataset')
    phenotypes = traits.Dict(mandatory=True, dtype=float, desc='phenotype values')
    # embeddings = traits.Dict(mandatory=True, desc='embeddings for gradients')
    # params = traits.Dict(mandatory=True, desc='parameters for anatomical connectivity')
    cv_split = traits.Dict(mandatory=True, dtype=list, desc='test subjects of each fold')

    level = traits.Str(mandatory=True, desc='parcellation level (1 to 4)')
    repeat = traits.Int(mandatory=True, desc='current repeat of cross-validation')
    fold = traits.Int(mandatory=True, desc='current fold in the repeat')

    rw_ypred = traits.Dict(
        mandatory=True, desc='Predicted psychometric values from region-wise models')
    # selected_regions = traits.Dict(
    #    mandatory=True, dtype=list, desc='selected regions for each parcellation level')
    mw_ypred = traits.Dict(desc='Predicted psychometric values from modality-wise models')
    fw_ypred = traits.Dict(desc='Predicted psychometric values from feature-wise models')
    config = traits.Dict(mandatory=True, desc='configuration settings')


class _IntegratedFeaturesModelOutputSpec(TraitedSpec):
    results = traits.Dict(desc='accuracy results')


class IntegratedFeaturesModel(SimpleInterface):
    """Train and test the integrated features model"""
    input_spec = _IntegratedFeaturesModelInputSpec
    output_spec = _IntegratedFeaturesModelOutputSpec

    def _en(
            self, train_y: np.ndarray, test_y: np.ndarray, train_ypred: np.ndarray,
            test_ypred: np.ndarray, key: str) -> dict:
        r, cod, model = elastic_net(
            train_ypred, train_y, test_ypred, test_y, int(self.inputs.config['n_alphas']))
        results = {f'en_r_{key}': r, f'en_cod_{key}': cod}

        return results

    @staticmethod
    def _kr_corr(
            train_y: np.ndarray, test_y: np.ndarray, train_ypred: np.ndarray,
            test_ypred: np.ndarray, key: str) -> dict:
        r, cod = kernel_ridge_corr_cv(train_ypred, train_y, test_ypred, test_y)
        results = {f'krcorr_r_{key}': r, f'krcorr_cod_{key}': cod}

        return results

    def _run_interface(self, runtime):
        all_sub = sum(self.inputs.sublists.values(), [])
        test_sub = self.inputs.cv_split[f'repeat{self.inputs.repeat}_fold{self.inputs.fold}']
        train_sub = [subject for subject in all_sub if subject not in test_sub]
        key = f'repeat{self.inputs.repeat}_fold{self.inputs.fold}_level{self.inputs.level}'

        train_y = np.array([self.inputs.phenotypes[sub] for sub in train_sub])
        test_y = np.array([self.inputs.phenotypes[sub] for sub in test_sub])
        train_ypred = np.hstack((
            self.inputs.rw_ypred['train_ypred'], self.inputs.mw_ypred['train_ypred'],
            self.inputs.fw_ypred['train_ypred']))
        test_ypred = np.hstack((
            self.inputs.rw_ypred['test_ypred'], self.inputs.mw_ypred['test_ypred'],
            self.inputs.fw_ypred['test_ypred']))

        en = self._en(train_y, test_y, train_ypred, test_ypred, key)
        kr_corr = self._kr_corr(train_y, test_y, train_ypred, test_ypred, key)
        self._results['results'] = en | kr_corr

        return runtime
