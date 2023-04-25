import itertools
from pathlib import Path

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.stats.multitest import multipletests

from mpp.utilities.models import elastic_net, permutation_test
from mpp.utilities.data import read_h5, write_h5
from mpp.utilities.features import diffusion_mapping_sub, score_sub


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
    confounds = traits.Dict(mandatory=True, dtype=dict, desc='confound values')
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

    def _extract_data(
            self, subjects: list, repeat: int, permutation: bool = False) -> tuple[np.ndarray, ...]:
        confounds = [
            'age', 'gender', 'handedness', 'brainseg_vol', 'icv_vol',
            'age2', 'ageGender', 'age2Gender']
        y = np.zeros((len(subjects)))
        x_all = np.zeros((len(subjects), len(confounds)))

        for i, subject in enumerate(subjects):
            if self.inputs.level == 'conf':
                for j, confound in enumerate(confounds):
                    x_all[i, j] = self.inputs.confounds[confound][subject]
            else:
                dataset = [
                    key for key in self.inputs.sublists if subject in self.inputs.sublists[key]][0]
                if dataset == 'HCP-A' or 'HCP-D':
                    feature_file = Path(
                        self.inputs.features_dir[dataset], f'{dataset}_{subject}_V1_MR.h5')
                else:
                    feature_file = Path(
                        self.inputs.features_dir[dataset], f'{dataset}_{subject}.h5')

                rsfc = read_h5(feature_file, f'/rsfc/level{self.inputs.level}')
                dfc = read_h5(feature_file, f'/dfc/level{self.inputs.level}')
                strength = read_h5(
                    feature_file, f'/network_stats/strength/level{self.inputs.level}')
                betweenness = read_h5(
                    feature_file, f'/network_stats/betweenness/level{self.inputs.level}')
                participation = read_h5(
                    feature_file, f'/network_stats/participation/level{self.inputs.level}')
                efficiency = read_h5(
                    feature_file, f'/network_stats/efficiency/level{self.inputs.level}')
                # tfc = read_h5(feature_file, f'/tfc/level{self.inputs.level}')
                myelin = read_h5(feature_file, f'/myelin/level{self.inputs.level}')
                gmv = read_h5(feature_file, f'/morphometry/GMV/level{self.inputs.level}')
                cs = read_h5(feature_file, f'/morphometry/CS/level{self.inputs.level}')
                ct = read_h5(feature_file, f'/morphometry/CT/level{self.inputs.level}')

                if permutation:
                    gradients = diffusion_mapping_sub(
                        self.inputs.embeddings[f'repeat{repeat}'], rsfc)
                    ac_gmv = score_sub(self.inputs.params[f'repeat{repeat}'], gmv)
                    ac_cs = score_sub(self.inputs.params[f'repeat{repeat}'], cs)
                    ac_ct = score_sub(self.inputs.params[f'repeat{repeat}'], ct)
                else:
                    gradients = diffusion_mapping_sub(self.inputs.embeddings['embedding'], rsfc)
                    ac_gmv = score_sub(self.inputs.params['params'], gmv)
                    ac_cs = score_sub(self.inputs.params['params'], cs)
                    ac_ct = score_sub(self.inputs.params['params'], ct)

                x = np.vstack((
                    rsfc.mean(axis=2), dfc.mean(axis=2),
                    strength, betweenness, participation, efficiency,
                    # tfc.reshape(tfc.shape[0], tfc.shape[1]*tfc.shape[2]).T,
                    myelin, gmv,
                    np.pad(cs, (0, len(gmv)-len(cs))), np.pad(ct, (0, len(gmv)-len(ct))),
                    gradients, ac_gmv,
                    np.hstack((ac_cs, np.zeros((ac_cs.shape[0], len(gmv)-len(cs))))),
                    np.hstack((ac_ct, np.zeros((ac_cs.shape[0], len(gmv)-len(ct)))))))

                # TODO: diffusion features

                x_all = x if i == 0 else np.dstack((x_all.T, x.T)).T

            # phenotype
            if permutation:
                y[i] = self.inputs.phenotypes_perm[subjects[i]]
            else:
                y[i] = self.inputs.phenotypes[subjects[i]]

        return x_all, y

    def _validate(self, repeat: int, permutation: bool = False) -> tuple[np.ndarray, ...]:
        if permutation:
            cv_split = self.inputs.cv_split_perm
        else:
            cv_split = self.inputs.cv_split
        n_folds = int(self.inputs.config["n_folds"])
        all_sub = sum(self.inputs.sublists.values(), [])

        test_sub = cv_split[f'repeat{repeat}_fold{self.inputs.fold}']
        val_sub = cv_split[f'repeat{repeat}_fold{(self.inputs.fold+1)%n_folds}']
        testval_sub = np.concatenate((val_sub, test_sub))
        train_sub = [subject for subject in all_sub if subject not in testval_sub]

        train_x, train_y = self._extract_data(train_sub, repeat, permutation=permutation)
        val_x, val_y = self._extract_data(val_sub, repeat, permutation=permutation)

        r = np.zeros((train_x.shape[2]))
        cod = np.zeros((train_x.shape[2]))
        for region in range(train_x.shape[2]):
            r[region], cod[region], _ = elastic_net(
                train_x[:, :, region], train_y, val_x[:, :, region], val_y,
                int(self.inputs.config['n_alphas']))
        r = np.nan_to_num(r)

        return r, cod

    def _test(self) -> tuple[np.ndarray, ...]:
        all_sub = sum(self.inputs.sublists.values(), [])
        test_sub = self.inputs.cv_split[f'repeat{self.inputs.repeat}_fold{self.inputs.fold}']
        train_sub = [subject for subject in all_sub if subject not in test_sub]

        train_x, train_y = self._extract_data(train_sub, self.inputs.repeat)
        test_x, test_y = self._extract_data(test_sub, self.inputs.repeat)

        r = np.zeros((train_x.shape[2]))
        cod = np.zeros((train_x.shape[2]))
        coef = np.zeros((train_x.shape[2], train_x.shape[1]))
        for region in range(train_x.shape[2]):
            if self.inputs.selected[f'regions_level{self.inputs.level}'][region]:
                r[region], cod[region], coef[region, :] = elastic_net(
                    train_x[:, :, region], train_y, test_x[:, :, region], test_y,
                    int(self.inputs.config['n_alphas']))

        return r, cod, coef

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
            r, cod, coef = self._test()
            self._results['results'][f'r_{key}'] = r
            self._results['results'][f'cod_{key}'] = cod
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

    def _save_results(self, results_dict: dict) -> None:
        Path(self.inputs.output_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(self.inputs.output_dir, 'regionwise_results.h5')
        for key in results_dict:
            write_h5(output_file, f'/{key}', np.array(results_dict[key]), self.inputs.overwrite)

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

        self._save_results(self._results['selected'])

        return runtime
