from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import itertools
import numpy as np
from os import path
import logging

from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.stats.multitest import multipletests

from mpp.utilities.models import elastic_net, permutation_test

base_dir = path.join(path.dirname(path.realpath(__file__)), '..')
logging.getLogger('datalad').setLevel(logging.WARNING)

task_runs = {'HCP-YA': ['tfMRI_EMOTION_LR', 'tfMRI_EMOTION_RL', 'tfMRI_GAMBLING_LR', 'tfMRI_GAMBLING_RL',
                        'tfMRI_LANGUAGE_LR', 'tfMRI_LANGUAGE_RL', 'tfMRI_MOTOR_LR', 'tfMRI_MOTOR_RL',
                        'tfMRI_RELATIONAL_LR', 'tfMRI_RELATIONAL_RL', 'tfMRI_SOCIAL_LR', 'tfMRI_SOCIAL_RL',
                        'tfMRI_WM_LR', 'tfMRI_WM_RL'],
            'HCP-A': ['tfMRI_CARIT_PA', 'tfMRI_FACENAME_PA', 'tfMRI_VISMOTOR_PA'],
            'HCP-D': ['tfMRI_CARIT_AP', 'tfMRI_CARIT_PA', 'tfMRI_EMOTION_PA', 'tfMRI_GUESSING_AP',
                      'tfMRI_GUESSING_PA']}

### CrossValSplit: generate cross-validation splits

class _CrossValSplitInputSpec(BaseInterfaceInputSpec):
    sublists = traits.Dict(mandatory=True, dtype=list, desc='list of subjects available in each dataset')
    config = traits.Dict(usedefault=True, desc='configuration settings')
    permutation = traits.Bool(False, desc='use n_repeats_perm instead')

class _CrossValSplitOutputSpec(TraitedSpec):
    cv_split = traits.Dict(dtype=list, desc='list of subjects in the test split of each fold')

class CrossValSplit(SimpleInterface):
    input_spec = _CrossValSplitInputSpec
    output_spec = _CrossValSplitOutputSpec

    def _run_interface(self, runtime):
        subjects = sum(self.inputs.sublists.values(), [])
        datasets = [[dataset] * len(self.inputs.sublists[dataset]) for dataset in self.inputs.sublists]
        datasets = itertools.chain.from_iterable(datasets)

        self._results['cv_split'] = {}
        if self.inputs.permutation:
            n_repeats = self.inputs.config['n_repeats_perm']
        else:
            n_repeats = self.inputs.config['n_repeats']
        n_folds = self.inputs.config['n_folds']
        rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, 
                                       random_state=self.inputs.config['cv_seed'])
        for fold, (_, test_ind) in enumerate(rskf.split(subjects, datasets)):
            key = f'repeat{int(np.floor(fold/n_folds))}_fold{int(fold%n_folds)}'
            self._results['cv_split'][key] = subjects[test_ind]

        return runtime

### RegionwiseModel: train and validate a region-wise model

class _RegionwiseModelInputSpec(BaseInterfaceInputSpec):
    sublists = traits.Dict(mandatory=True, dtype=list, desc='list of subjects available in each dataset')
    image_features = traits.Dict(mandatory=True, dtype=dict, desc='previously extracted imaging features')
    confounds = traits.Dict(mandatory=True, dtype=dict, desc='confound values from subjects in sublists')
    level = traits.Str(mandatory=True, desc='parcellation level (1 to 4) or conf for confounder model')
    region = traits.Int(mandatory=True, desc='region number within the parcellation level (0 to P-1)')
    mode = traits.Str(mandatory=True, desc=("'validate' to train and validate models with permutation tests, "
                                            + "'test' to train and test models on test folds"))
    config = traits.Dict(usedefault=True, desc='configuration settings')

    cv_split = traits.Dict(mandatory=True, dtype=list, desc='list of subjects in the test split of each fold')
    phenotypes = traits.Dict(mandatory=True, dtype=float, desc='phenotype values from subjects in sublists')
    gradients = traits.Dict(mandatory=True, dtype=dict, desc='gradient loading features')
    ac = traits.Dict(mandatory=True, dtype=dict, desc='gradient loading features')

    cv_split_perm = traits.Dict(usedefault=True, dtype=list, desc='list of subjects in the test split of each fold')
    phenotypes_perm = traits.Dict(usedefault=True, dtype=float, desc='shuffled phenotype values for permutation')
    gradients_perm = traits.Dict(usedefault=True, dtype=dict, desc='gradient loading features for permutation splits')
    ac_perm = traits.Dict(usedefault=True, type=dict, desc='gradient loading features for permutation splits')

class _RegionwiseModelOutputSpec(TraitedSpec):
    r = traits.Float(desc='correlation accuracy')
    cod = traits.Float(desc='coefficient of determination accuracy')
    p_r = traits.Float(usedefault=True, desc='permutation test p value based on correlation accuracy')
    p_cod = traits.Float(usedefault=True, desc='permutation test p values based on COD accuracy')
    selected = traits.List(usedefault=True, dtype=bool, desc='whether each feature is selected or not')

class RegionwiseModel(SimpleInterface):
    input_spec = _RegionwiseModelInputSpec
    output_spec = _RegionwiseModelOutputSpec

    def _extract_data(self, subjects, repeat, fold, permutation=False):
        confounds = ['age', 'gender', 'handedness', 'brainseg_vol', 'icv_vol', 'age2', 'ageGender', 'age2Gender']
        y = np.zeros((len(subjects)))
        x_all = np.zeros((len(subjects), len(confounds)))

        for i in range(len(subjects)):
            fdict = self.inputs.image_feautres[subjects[i]]

            if self.inputs.level == 'conf':
                for j in range(len(confounds)):
                    x_all[i, j] = self.inputs.confounds[confounds[j]][subjects[i]]
            else:
                # resting-state features
                x = fdict[f'rsfc_level{self.inputs.level}'].mean(axis=2)[:, self.inputs.region]
                x = np.concatenate([x, fdict[f'dfc_level{self.inputs.level}'].mean(axis=2)[:, self.inputs.region]])
                for stats in ['strength', 'betweenness', 'participation', 'efficiency']:
                    x = np.concatenate([x, [fdict[f'{stats}_level{self.inputs.level}'][self.inputs.region]]])
                gradient_key = f'repeat{repeat}_fold{fold}_level{self.inputs.level}'
                if permutation:
                    x = np.concatenate([x, self.inputs.gradients_perm[subjects[i]][gradient_key][:, self.inputs.region]])
                else:
                    x = np.concatenate([x, self.inputs.gradients[subjects[i]][gradient_key][:, self.inputs.region]])

                # task features
                dataset = [key for key in self.inputs.sublists if 'sub-04' in self.inputs.sublists[key]][0]
                for task in task_runs[dataset]:
                    x = np.concatenate([x, fdict[f'tfc_{task}_level{self.inputs.level}']])

                # structural features
                if self.inputs.region < ((self.inputs.level) * 100):
                    stats_struct = ['GMV', 'CS', 'CT', 'myelin'] # cortical region features
                else:
                    stats_struct = ['GMV', 'myelin'] # subcortical region features
                for stats in stats_struct:
                    x = np.concatenate([x, [fdict[f'{stats}_level{self.inputs.level}'][self.inputs.region]]])
                    ac_key = f'repeat{repeat}_fold{fold}_{stats}_level{self.inputs.level}'
                    if permutation:
                        x = np.concatenate([x, self.inputs.ac_perm[subjects[i]][ac_key][:, self.inputs.region]])
                    else:
                        x = np.concatenate([x, self.inputs.ac[subjects[i]][ac_key][:, self.inputs.region]])

                # TODO: diffusion features

                x_all = x if i == 0 else np.vstack((x_all, x))

            # phenotype
            if permutation:
                y[i] = self.inputs.phenotypes_perm[subjects[i]]
            else:
                y[i] = self.inputs.phenotypes[subjects[i]]

        return x_all, y

    def _validate(self, permutation=False):
        if permutation:
            n_repeats = self.inputs.config['n_repeats_perm']
            cv_split = self.inputs.cv_split_perm
        else:
            n_repeats = self.inputs.config['n_repeats']
            cv_split = self.inputs.cv_split

        all_sub = sum(self.inputs.sublists.values(), [])
        all_r = np.zeros((n_repeats,))
        all_cod = np.zeros((n_repeats,))

        for repeat in range(len(n_repeats)):
            for fold in range(len(self.inputs.config['n_folds'])):
                test_sub = cv_split[f'repeat{repeat}_fold{fold}']
                val_sub = cv_split[f'repeat{repeat}_fold{(fold+1)%self.inputs.config["n_folds"]}']
                train_sub = [subject for subject in all_sub if subject not in (test_sub + val_sub)]

                train_x, train_y = self._extract_data(train_sub, repeat, fold, permutation=permutation)
                val_x, val_y = self._extract_data(val_sub, repeat, fold, permutation=permutation)
                r, cod = elastic_net(train_x, train_y, val_x, val_y, self.inputs.config['n_alphas'])

                all_r[repeat] = all_r[repeat] + np.nan_to_num(r)
                all_cod[repeat] = all_cod[repeat] + cod

            all_r[repeat] = all_r[repeat] / self.inputs.config['n_fold']
            all_cod[repeat] = all_cod[repeat] / self.inputs.config['n_fold']

        return all_r, all_cod

    def _test(self):
        all_sub = sum(self.inputs.sublists.values(), [])
        r = np.zeros((self.inputs.config['n_repeat'], self.inputs.config['n_folds']))
        cod = np.zeros((self.inputs.config['n_repeat'], self.inputs.config['n_folds']))
        selected = None

        for repeat in range(len(self.inputs.config['n_repeats'])):
            for fold in range(len(self.inputs.config['n_folds'])):
                test_sub = self.inputs.cv_split[f'repeat{repeat}_fold{fold}']
                train_sub = [subject for subject in all_sub if subject not in test_sub]

                train_x, train_y = self._extract_data(train_sub, repeat, fold)
                test_x, test_y = self._extract_data(test_sub, repeat, fold)
                r[repeat, fold], cod[repeat, fold], coef = elastic_net(train_x, train_y, test_x, test_y, 
                                                                       self.inputs.config['n_alphas'])
                if selected == None:
                    selected = (coef != 0)
                else:
                    selected = [a or b for a, b in zip(selected, (coef != 0))]

        return r.mean(), cod.mean(), selected

    def _run_interface(self, runtime):
        if self.inputs.mode == 'validate':
            r, cod = self._validate()
            perm_r, perm_cod = self._validate(permutation=True)

            self._results['r'] = r.mean()
            self._results['cod'] = cod.mean()
            self._results['p_r'] = permutation_test(self._results['r'], perm_r)
            self._results['p_cod'] = permutation_test(self._results['cod'], perm_cod)
        
        elif self.inputs.mode == 'test':
            self._results['r'], self._results['cod'], self._results['selected'] = self._test() 

        return runtime

### FeatureSelect: extract features and available phenotype data

class _FeatureSelectInputSpec(BaseInterfaceInputSpec):
    p_r = traits.List(mandatory=True, dtype=float, desc='permutation test p value based on correlation accuracy')
    p_cod = traits.List(mandatory=True, dtype=float, desc='permutation test p values based on COD accuracy')
    criteria = traits.Str('r', desc='which accuracy to use for selection (r or cod)')
    levels = traits.List(mandatory=True, dtype=str, desc='parcellation levels corresponding to the region numbers')
    regions = traits.List(mandatory=True, dtype=int, desc='region number within each parcellation level')

class _FeatureSelectOutputSpec(TraitedSpec):
    selected_levels = traits.List(dtype=str, desc='parcellation levels corresponding to the region numbers')
    selected_regions = traits.List(dtype=int, desc='region number within each parcellation level')

class FeatureSelect(SimpleInterface):
    input_spec = _FeatureSelectInputSpec
    output_spec = _FeatureSelectOutputSpec

    def _run_interface(self, runtime):
        p = self.inputs.p_r if self.inputs.criteria == 'r' else self.inputs.p_cod
        selected = multipletests(p, method='fdr_bh')[0]
        self._results['selected_levels'] = list(itertools.compress(self.inputs.levels, selected))
        self._results['selected_regions'] = list(itertools.compress(self.inputs.regions))

        return runtime