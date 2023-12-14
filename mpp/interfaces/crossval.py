import itertools

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from mpp.utilities.models import elastic_net, random_forest_cv
from mpp.utilities.data import cv_extract_subject_data
from mpp.utilities.features import pheno_reg_conf


class _CrossValSplitInputSpec(BaseInterfaceInputSpec):
    sublists = traits.Dict(mandatory=True, dtype=list, desc='availabel subjects in each dataset')
    config = traits.Dict(mandatory=True, desc='configuration settings')


class _CrossValSplitOutputSpec(TraitedSpec):
    cv_split = traits.Dict(dtype=list, desc='list of subjects in the training split of each fold')


class CrossValSplit(SimpleInterface):
    """Generate cross-validation splits"""
    input_spec = _CrossValSplitInputSpec
    output_spec = _CrossValSplitOutputSpec

    def _run_interface(self, runtime):
        subjects = sum(self.inputs.sublists.values(), [])
        datasets = [[ds] * len(self.inputs.sublists[ds]) for ds in self.inputs.sublists]
        datasets = list(itertools.chain.from_iterable(datasets))

        self._results['cv_split'] = {}
        n_repeats = int(self.inputs.config['n_repeats'])
        n_folds = int(self.inputs.config['n_folds'])
        rskf = RepeatedStratifiedKFold(
            n_splits=n_folds, n_repeats=n_repeats, random_state=int(self.inputs.config['cv_seed']))
        for fold, (train_ind, _) in enumerate(rskf.split(subjects, datasets)):
            key = f'repeat{int(np.floor(fold / n_folds))}_fold{int(fold % n_folds)}'
            train_sub = subjects[train_ind]
            self._results['cv_split'][key] = train_sub

            skf = StratifiedKFold(n_splits=5)
            for inner, (train_i, test_i) in enumerate(skf.split(train_sub, datasets[train_ind])):
                key_inner = f'{key}_inner{inner}'
                self._results['cv_split'][key_inner] = train_sub[train_i]
                self._results['cv_split'][f'{key_inner}_train'] = train_i
                self._results['cv_split'][f'{key_inner}_test'] = test_i

        return runtime


class _FeaturewiseModelInputSpec(BaseInterfaceInputSpec):
    sublists = traits.Dict(mandatory=True, dtype=list, desc='available subjects in each dataset')
    features_dir = traits.Dict(
        mandatory=True, dtype=str, desc='absolute path to extracted features for each dataset')
    phenotypes = traits.Dict(mandatory=True, dtype=float, desc='phenotype values')
    confounds = traits.Dict(dtype=dict, desc='confound values from subjects in sublists')
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
        'rsfc', 'dfc', 'efc', 'gradients', 'tfc', 'strength', 'betweenness', 'participation',
        'efficiency', 'myelin', 'gmv', 'cs', 'ct', 'ac_gmv', 'ac_cs', 'ac_ct', 'scc', 'scl',
        'scc_strength', 'scc_betweenness', 'scc_participation', 'scc_efficiency', 'scl_strength',
        'scl_betweenness', 'scl_participation', 'scl_efficiency']

    @staticmethod
    def _add_sub_data(data_dict, sub_data, key, ind):
        if ind == 0:
            data_dict[key] = sub_data
        else:
            data_dict[key] = np.vstack((data_dict[key], sub_data))
        return data_dict

    def _extract_data(self, subjects: list) -> tuple[dict, np.ndarray, np.ndarray]:
        y = np.zeros(len(subjects))
        confounds = np.zeros((len(subjects), len(list(self.inputs.confounds.keys()))))
        x_all = {}

        for i, subject in enumerate(subjects):
            x = cv_extract_subject_data(
                self.inputs.sublists, subject, self.inputs.features_dir, self.inputs.level,
                False, self.inputs.embeddings, self.inputs.params,
                self.inputs.repeat)
            for key, x_curr in zip(self._feature_list, x):
                if key in ['rsfc', 'dfc', 'efc', 'ac_gmv', 'ac_cs', 'ac_ct', 'scc', 'scl']:
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
            for j, conf in enumerate(list(self.inputs.confounds.keys())):
                confounds[i, j] = self.inputs.confounds[conf][subjects[i]]
            y[i] = self.inputs.phenotypes[subjects[i]]

        return x_all, y, confounds

    def _run_interface(self, runtime):
        self._results['results'] = {}
        key = f'repeat{self.inputs.repeat}_fold{self.inputs.fold}'
        all_sub = sum(self.inputs.sublists.values(), [])
        train_sub = self.inputs.cv_split[key]
        test_sub = [subject for subject in all_sub if subject not in train_sub]

        train_x, train_y, train_conf = self._extract_data(train_sub)
        test_x, test_y, test_conf = self._extract_data(test_sub)
        train_y, test_y = pheno_reg_conf(train_y, train_conf, test_y, test_conf)
        for i in range(len(train_x.keys())-len(self._feature_list)):
            self._feature_list.insert(4+i, f'tfc_{i+1}')

        train_ypred = np.zeros((len(train_sub), len(self._feature_list)))
        test_ypred = np.zeros((len(test_sub), len(self._feature_list)))
        train_cod = np.zeros(len(self._feature_list))
        iters = zip(train_x.values(), test_x.values(), enumerate(self._feature_list))
        for train_curr, test_curr, (i, feature) in iters:
            key_out = [f'{feature}_{key}_level{self.inputs.level}']
            for inner in range(5):
                train_ind = self.inputs.cv_split[f'{key}_inner{inner}_train']
                test_ind = self.inputs.cv_split[f'{key}_inner{inner}_test']
                _, cod, train_pred, _= elastic_net(
                    train_curr[train_ind, :], train_y[train_ind], train_curr[test_ind, :],
                    train_y[test_ind], int(self.inputs.config['n_alphas']))
                train_ypred[test_ind, i] = train_pred

            r, cod, test_pred, train_cod_curr = elastic_net(
                train_curr, train_y, test_curr, test_y, int(self.inputs.config['n_alphas']))
            test_ypred[:, i] = test_pred
            train_cod[i] = train_cod_curr
            self._results['results'][f'en_r_{key_out}'] = r
            self._results['results'][f'en_cod_{key_out}'] = cod

        self._results['results']['fw_ypred'] = {
            'train_ypred': train_ypred, 'test_ypred': test_ypred, 'train_cod': train_cod}

        return runtime


class _ConfoundsModelInputSpec(BaseInterfaceInputSpec):
    sublists = traits.Dict(mandatory=True, dtype=list, desc='available subjects in each dataset')
    phenotypes = traits.Dict(mandatory=True, dtype=float, desc='phenotype values')
    confounds = traits.Dict(dtype=dict, desc='confound values from subjects in sublists')
    cv_split = traits.Dict(mandatory=True, dtype=list, desc='test subjects of each fold')
    repeat = traits.Int(mandatory=True, desc='current repeat of cross-validation')
    fold = traits.Int(mandatory=True, desc='current fold in the repeat')
    config = traits.Dict(mandatory=True, desc='configuration settings')


class _ConfoundsModelOutputSpec(TraitedSpec):
    results = traits.Dict(desc='accuracy results')
    c_ypred = traits.Dict(desc='Predicted psychometric values')


class ConfoundsModel(SimpleInterface):
    """Train and test confound models"""
    input_spec = _ConfoundsModelInputSpec
    output_spec = _ConfoundsModelOutputSpec

    def _extract_data(self, subjects: list) -> tuple[np.ndarray, np.ndarray]:
        y = np.zeros(len(subjects))
        x = np.zeros((len(subjects), len(list(self.inputs.confounds.keys()))))
        for i, _ in enumerate(subjects):
            for j, conf in enumerate(list(self.inputs.confounds.keys())):
                x[i, j] = self.inputs.confounds[conf][subjects[i]]
            y[i] = self.inputs.phenotypes[subjects[i]]

        return x, y

    def _run_interface(self, runtime):
        self._results['results'] = {}
        key = f'repeat{self.inputs.repeat}_fold{self.inputs.fold}'
        conf_list = list(self.inputs.confounds.keys())
        all_sub = sum(self.inputs.sublists.values(), [])
        train_sub = self.inputs.cv_split[key]
        test_sub = [subject for subject in all_sub if subject not in train_sub]

        train_x, train_y = self._extract_data(train_sub)
        test_x, test_y = self._extract_data(test_sub)

        train_ypred = np.zeros((len(train_sub), len(conf_list)))
        test_ypred = np.zeros((len(test_sub), len(conf_list)))
        train_cod = np.zeros(len(conf_list))
        for i, conf in enumerate(conf_list):
            key_out = [f'{conf}_{key}']
            for inner in range(5):
                train_ind = self.inputs.cv_split[f'{key}_inner{inner}_train']
                test_ind = self.inputs.cv_split[f'{key}_inner{inner}_test']
                _, cod, train_pred, _ = elastic_net(
                    train_x[train_ind, i].reshape(-1, 1), train_y[train_ind],
                    train_x[test_ind, i].reshape(-1, 1), train_y[test_ind],
                    int(self.inputs.config['n_alphas']))
                train_ypred[test_ind, i] = train_pred

            r, cod, test_pred, train_cod_curr = elastic_net(
                train_x[:, i].reshape(-1, 1), train_y, test_x[:, i].reshape(-1, 1), test_y,
                int(self.inputs.config['n_alphas']))
            test_ypred[:, i] = test_pred
            train_cod[i] = train_cod_curr
            self._results['results'][f'en_r_{key_out}'] = r
            self._results['results'][f'en_cod_{key_out}'] = cod

        self._results['results']['c_ypred'] = {
            'train_ypred': train_ypred, 'test_ypred': test_ypred, 'train_cod': train_cod}

        return runtime


class _IntegratedFeaturesModelInputSpec(BaseInterfaceInputSpec):
    sublists = traits.Dict(mandatory=True, dtype=list, desc='available subjects in each dataset')
    phenotypes = traits.Dict(mandatory=True, dtype=float, desc='phenotype values')
    cv_split = traits.Dict(mandatory=True, dtype=list, desc='test subjects of each fold')
    config = traits.Dict(mandatory=True, desc='configuration settings')

    level = traits.Str(mandatory=True, desc='parcellation level (1 to 4)')
    repeat = traits.Int(mandatory=True, desc='current repeat of cross-validation')
    fold = traits.Int(mandatory=True, desc='current fold in the repeat')

    fw_ypred = traits.Dict(desc='Predicted psychometric values from feature-wise models')
    c_ypred = traits.Dict(desc='Predicted psychometric values from confound models')


class _IntegratedFeaturesModelOutputSpec(TraitedSpec):
    results = traits.Dict(desc='accuracy results')


class IntegratedFeaturesModel(SimpleInterface):
    """Train and test the integrated features model"""
    input_spec = _IntegratedFeaturesModelInputSpec
    output_spec = _IntegratedFeaturesModelOutputSpec

    def _rfr(
            self, train_y: np.ndarray, test_y: np.ndarray, train_ypred: np.ndarray,
            test_ypred: np.ndarray, key: str) -> None:
        r, cod, _ = random_forest_cv(train_ypred, train_y, test_ypred, test_y)
        self._results['results'][f'rfr_r_{key}'] = r
        self._results['results'][f'rfr_cod_{key}'] = cod

    def _run_interface(self, runtime):
        all_sub = sum(self.inputs.sublists.values(), [])
        train_sub = self.inputs.cv_split[f'repeat{self.inputs.repeat}_fold{self.inputs.fold}']
        test_sub = [subject for subject in all_sub if subject not in train_sub]

        train_y = np.array([self.inputs.phenotypes[sub] for sub in train_sub])
        test_y = np.array([self.inputs.phenotypes[sub] for sub in test_sub])
        train_ypred = np.hstack((
            self.inputs.fw_ypred['train_ypred'], self.inputs.c_ypred['train_ypred']))
        test_ypred = np.hstack((
            self.inputs.fw_ypred['test_ypred'], self.inputs.c_ypred['test_ypred']))
        train_cod = np.concatenate((
            self.inputs.fw_ypred['train_cod'], self.inputs.c_ypred['train_cod']))
        f_ranks = -np.argsort(-train_cod)

        self._results['results'] = {}
        for n_feature in range(2, train_ypred.shape[1]+1):
            key = f'repeat{self.inputs.repeat}_fold{self.inputs.fold}_level{self.inputs.level}' \
                  f'_{n_feature}features'
            self._rfr(
                train_y, test_y, train_ypred[:, f_ranks[:n_feature]],
                test_ypred[:, f_ranks[:n_feature]], key)

        return runtime
