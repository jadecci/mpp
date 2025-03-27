from pathlib import Path
import itertools

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
import numpy as np
import pandas as pd

from mpp.utilities import find_sub_file, fc_to_matrix, pheno_reg_conf, elastic_net


class _CrossValSplitInputSpec(BaseInterfaceInputSpec):
    sublists = traits.Dict(mandatory=True, dtype=list, desc="availabel subjects in each dataset")
    config = traits.Dict(mandatory=True, desc="workflow configurations")


class _CrossValSplitOutputSpec(TraitedSpec):
    cv_split = traits.Dict(dtype=list, desc="list of subjects in the training split of each fold")


class CrossValSplit(SimpleInterface):
    """Generate cross-validation splits"""
    input_spec = _CrossValSplitInputSpec
    output_spec = _CrossValSplitOutputSpec

    def _run_interface(self, runtime):
        subjects = sum(self.inputs.sublists.values(), [])
        datasets = [[ds] * len(self.inputs.sublists[ds]) for ds in self.inputs.sublists]
        datasets = list(itertools.chain.from_iterable(datasets))
        self._results["cv_split"] = {}
        n_repeats = int(self.inputs.config["n_repeats"])
        n_folds = int(self.inputs.config["n_folds"])

        if len(np.unique(datasets)) == 1 and np.unique(datasets)[0] == "HCP-YA":
            fam_id = pd.read_csv(self.inputs.config["hcpya_res"], usecols=["Subject", "Family_ID"])
            fam_id = fam_id.loc[fam_id["Subject"].isin(subjects)]
            rng = np.random.default_rng(seed=int(self.inputs.config["cv_seed"]))
            cv_iter = [[[], []] for i in range(n_repeats * n_folds)]
            fold_size_min = np.round(len(subjects) / n_folds)
            for repeat in range(n_repeats):
                ind_to_fill = np.arange(len(subjects))
                for fold in range(n_folds):
                    cv_ind = fold + repeat * n_folds
                    n_max = len(subjects) - (fold + 1) * fold_size_min
                    while len(ind_to_fill) > n_max and len(ind_to_fill):
                        fill_start = rng.integers(low=0, high=len(ind_to_fill))
                        fill_start_ind = ind_to_fill[fill_start]
                        cv_iter[cv_ind][1].append(fill_start_ind)
                        ind_to_fill = np.delete(ind_to_fill, fill_start)

                        fill_fam_id = fam_id["Family_ID"].iloc[fill_start_ind]
                        fill_fam = fam_id["Subject"].loc[
                            (fam_id["Family_ID"] == fill_fam_id) & (fam_id.index != fill_start_ind)]
                        for ind in fill_fam.index.to_list():
                            cv_iter[cv_ind][1].append(ind)
                            ind_to_fill = np.delete(ind_to_fill, np.where(ind_to_fill == ind))
                    cv_iter[cv_ind][0] = [
                        i for i in range(len(subjects)) if i not in cv_iter[cv_ind][1]]
        else:
            rskf = RepeatedStratifiedKFold(
                n_splits=n_folds, n_repeats=n_repeats,
                random_state=int(self.inputs.config["cv_seed"]))
            cv_iter = rskf.split(subjects, datasets)

        for fold, (train_ind, _) in enumerate(cv_iter):
            key = f"repeat{int(np.floor(fold / n_folds))}_fold{int(fold % n_folds)}"
            train_sub = np.array(subjects)[train_ind]
            self._results["cv_split"][key] = train_sub

            skf = StratifiedKFold(n_splits=5)
            inner_cv = enumerate(skf.split(train_sub, np.array(datasets)[train_ind]))
            for inner, (train_i, test_i) in inner_cv:
                key_inner = f"{key}_inner{inner}"
                self._results["cv_split"][key_inner] = train_sub[train_i]
                self._results["cv_split"][f"{key_inner}_test"] = test_i

        return runtime


class _FeaturewiseModelInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    feature_type = traits.Str(mandatory=True, desc="Feature type")
    target = traits.Str(mandatory=True, desc="target phenotype to predict")
    sublists = traits.Dict(mandatory=True, dtype=list, desc="available subjects in each dataset")
    cv_split = traits.Dict(mandatory=True, dtype=list, desc="test subjects of each fold")
    cv_features_file = traits.File(mandatory=True, desc="file containing CV features")
    repeat = traits.Int(mandatory=True, desc="current repeat of cross-validation")
    fold = traits.Int(mandatory=True, desc="current fold in the repeat")


class _FeaturewiseModelOutputSpec(TraitedSpec):
    results = traits.Dict(desc="accuracy results")
    fw_ypred = traits.Dict(desc="Predicted psychometric values")
    feature_type = traits.Str(mandatory=True, desc="Feature type")


class FeaturewiseModel(SimpleInterface):
    """Train and test feature-wise models"""
    input_spec = _FeaturewiseModelInputSpec
    output_spec = _FeaturewiseModelOutputSpec

    def _grad_feature(self, subject_file: Path, postfix: str) -> pd.DataFrame:
        nparc_dict = {"1": 116, "2": 232, "3": 350, "4": 454}
        rsfc_arr = pd.read_hdf(subject_file, f"rs_sfc_level{self.inputs.config['level']}")
        rsfc = fc_to_matrix(pd.DataFrame(rsfc_arr), nparc_dict[self.inputs.config["level"]])
        embed = pd.read_hdf(self.inputs.cv_features_file, f"embed{postfix}")
        grad = pd.DataFrame(np.array(embed @ rsfc).flatten()).T
        return grad

    def _ac_feature(self, subject_file: Path, postfix: str) -> pd.DataFrame:
        # see https://github.com/katielavigne/score/blob/main/score.py
        morph_feature = f"s_{self.inputs.feature_type.split('s_ac')[1]}"
        morph = pd.DataFrame(pd.read_hdf(
            subject_file, f"{morph_feature}_level{self.inputs.config['level']}"))
        params = pd.DataFrame(pd.read_hdf(
            self.inputs.cv_features_file, f"params_{morph_feature.split('s_')[1]}{postfix}"))
        ac = []
        for i in range(morph.shape[1]):
            for j in range(morph.shape[1]):
                params_curr = params[f"{i}_{j}"]
                morph_curr = morph[f"{morph_feature}_{i}"]
                ac.append((
                        params_curr[0] + params_curr[1] * morph_curr + params_curr[2]
                        * morph.mean(axis=1)).iloc[0])
        return pd.DataFrame(ac).T

    def _extract_data(self, subjects: list, postfix: str = "") -> tuple[np.ndarray, ...]:
        level = self.inputs.config['level']
        y = pd.DataFrame()
        conf = pd.DataFrame()
        x = pd.DataFrame()
        for subject in subjects:
            subject_file, dti_file, subject_id = find_sub_file(
                self.inputs.sublists, self.inputs.config["features_dir"], subject)
            if self.inputs.feature_type == "rs_grad":
                x_curr = self._grad_feature(subject_file, postfix)
            elif self.inputs.feature_type in ["s_acgmv", "s_accs", "s_acct"]:
                x_curr = self._ac_feature(subject_file, postfix)
            elif self.inputs.feature_type == "rs_stats":
                x_cpl = pd.DataFrame(pd.read_hdf(subject_file, f"rs_cpl_level{level}"))
                x_eff = pd.DataFrame(pd.read_hdf(subject_file, f"rs_eff_level{level}"))
                x_mod = pd.DataFrame(pd.read_hdf(subject_file, f"rs_mod_level{level}"))
                x_par = pd.DataFrame(pd.read_hdf(subject_file, f"rs_par_level{level}"))
                x_curr = pd.concat([x_cpl, x_eff, x_mod, x_par], axis="columns")
            elif self.inputs.feature_type in ["d_fa", "d_md", "d_ad", "d_rd"]:
                feature_curr = self.inputs.feature_type.split("d_")[1]
                x_curr = pd.DataFrame(pd.read_hdf(dti_file, f"{feature_curr}_{subject_id}")).T
            else:
                x_curr = pd.DataFrame(pd.read_hdf(
                    subject_file, f"{self.inputs.feature_type}_level{level}"))
            x_curr = x_curr.replace(-np.inf, 0)
            x_curr = x_curr.fillna(value=0)
            x = pd.concat([x, x_curr], axis="index")
            y_curr = pd.DataFrame(pd.read_hdf(subject_file, "phenotype"))
            y = pd.concat([y, y_curr[self.inputs.target]], axis="index")
            conf_curr = pd.DataFrame(pd.read_hdf(subject_file, "confound"))
            conf = pd.concat([conf, conf_curr], axis="index")
        return x.to_numpy(), y.to_numpy(), conf.to_numpy()

    def _run_interface(self, runtime):
        key = f"repeat{self.inputs.repeat}_fold{self.inputs.fold}"
        key_out = f"{self.inputs.feature_type}_{key}_level{self.inputs.config['level']}"
        n_alphas = int(self.inputs.config["n_alphas"])
        self._results["feature_type"] = self.inputs.feature_type

        all_sub = sum(self.inputs.sublists.values(), [])
        train_sub = self.inputs.cv_split[key]
        test_sub = [subject for subject in all_sub if subject not in train_sub]
        train_x, train_y, train_conf = self._extract_data(train_sub)
        test_x, test_y, test_conf = self._extract_data(test_sub)
        train_y, test_y = pheno_reg_conf(train_y, train_conf, test_y, test_conf)
        r, cod, test_ypred = elastic_net(train_x, train_y, test_x, test_y, n_alphas)
        self._results["results"] = {
            f"r_{key_out}": r, f"cod_{key_out}": cod, f"test_ypred_{key_out}": test_ypred}

        train_ypred = np.zeros(len(train_sub))
        for inner in range(5):
            inner_train_sub = self.inputs.cv_split[f"{key}_inner{inner}"]
            inner_test_i = self.inputs.cv_split[f"{key}_inner{inner}_test"]
            inner_test_sub = train_sub[inner_test_i]
            train_x, train_y, train_conf = self._extract_data(
                inner_train_sub, postfix=f"_inner{inner}")
            test_x, test_y, test_conf = self._extract_data(inner_test_sub, postfix=f"_inner{inner}")
            train_y, test_y = pheno_reg_conf(train_y, train_conf, test_y, test_conf)
            _, _, train_ypred[inner_test_i] = elastic_net(
                train_x, train_y, test_x, test_y, n_alphas)
        self._results["results"].update({f"train_ypred_{key_out}": train_ypred})
        self._results["fw_ypred"] = {"train_ypred": train_ypred, "test_ypred": test_ypred}

        return runtime


class _ConfoundsModelInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    target = traits.Str(mandatory=True, desc="target phenotype to predict")
    sublists = traits.Dict(mandatory=True, dtype=list, desc="available subjects in each dataset")
    cv_split = traits.Dict(mandatory=True, dtype=list, desc="test subjects of each fold")
    repeat = traits.Int(mandatory=True, desc="current repeat of cross-validation")
    fold = traits.Int(mandatory=True, desc="current fold in the repeat")


class _ConfoundsModelOutputSpec(TraitedSpec):
    results = traits.Dict(desc='accuracy results')
    c_ypred = traits.Dict(desc='Predicted psychometric values')


class ConfoundsModel(SimpleInterface):
    """Train and test confound models"""
    input_spec = _ConfoundsModelInputSpec
    output_spec = _ConfoundsModelOutputSpec

    def _extract_data(self, subjects: list) -> tuple[np.ndarray, ...]:
        y = pd.DataFrame()
        conf = pd.DataFrame()
        for subject in subjects:
            subject_file, _, _ = find_sub_file(
                self.inputs.sublists, self.inputs.config["features_dir"], subject)
            y_curr = pd.DataFrame(pd.read_hdf(subject_file, "phenotype"))
            y = pd.concat([y, y_curr[self.inputs.target]], axis="index")
            conf_curr = pd.DataFrame(pd.read_hdf(subject_file, "confound"))
            conf = pd.concat([conf, conf_curr], axis="index")
        return conf.to_numpy(), y.to_numpy()

    def _run_interface(self, runtime):
        key = f"repeat{self.inputs.repeat}_fold{self.inputs.fold}"
        key_out = f"confound_{key}"
        n_alphas = int(self.inputs.config["n_alphas"])

        all_sub = sum(self.inputs.sublists.values(), [])
        train_sub = self.inputs.cv_split[key]
        test_sub = [subject for subject in all_sub if subject not in train_sub]
        train_x, train_y = self._extract_data(train_sub)
        test_x, test_y = self._extract_data(test_sub)
        r, cod, test_ypred = elastic_net(train_x, train_y, test_x, test_y, n_alphas)
        self._results["results"] = {
            f"r_{key_out}": r, f"cod_{key_out}": cod, f"test_ypred_{key_out}": test_ypred}

        train_ypred = np.zeros(len(train_sub))
        for inner in range(5):
            inner_train_sub = self.inputs.cv_split[f"{key}_inner{inner}"]
            inner_test_i = self.inputs.cv_split[f"{key}_inner{inner}_test"]
            inner_test_sub = train_sub[inner_test_i]
            train_x, train_y = self._extract_data(inner_train_sub)
            test_x, test_y = self._extract_data(inner_test_sub)
            _, _, train_ypred[inner_test_i] = elastic_net(
                train_x, train_y, test_x, test_y, n_alphas)
        self._results["results"].update({f"train_ypred_{key_out}": train_ypred})
        self._results["c_ypred"] =  {"train_ypred": train_ypred, "test_ypred": test_ypred}

        return runtime


class _IntegratedFeaturesModelInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    target = traits.Str(mandatory=True, desc="target phenotype to predict")
    sublists = traits.Dict(mandatory=True, dtype=list, desc="available subjects in each dataset")
    cv_split = traits.Dict(mandatory=True, dtype=list, desc="test subjects of each fold")
    repeat = traits.Int(mandatory=True, desc="current repeat of cross-validation")
    fold = traits.Int(mandatory=True, desc="current fold in the repeat")
    fw_ypred = traits.List(dtype=dict, mandatory=True, desc="Feature-wise predicted values")
    features = traits.List(dtype=str, mandatory=True, desc="feature types")
    c_ypred = traits.Dict(mandatory=True, desc="Confound predicted values")


class _IntegratedFeaturesModelOutputSpec(TraitedSpec):
    results = traits.Dict(desc="accuracy results")


class IntegratedFeaturesModel(SimpleInterface):
    """Train and test the integrated features model"""
    input_spec = _IntegratedFeaturesModelInputSpec
    output_spec = _IntegratedFeaturesModelOutputSpec

    def _extract_data(self, subjects: list, key: str) -> tuple[np.ndarray, ...]:
        y = pd.DataFrame()
        for subject in subjects:
            subject_file, _, _ = find_sub_file(
                self.inputs.sublists, self.inputs.config["features_dir"], subject)
            y_curr = pd.DataFrame(pd.read_hdf(subject_file, "phenotype"))
            y = pd.concat([y, y_curr[self.inputs.target]], axis="index")

        x = self.inputs.c_ypred[key]
        for fw_ypred in self.inputs.fw_ypred:
            x = np.vstack((x, fw_ypred[key]))
        return x.T, y.to_numpy()

    def _train_ranks(self, train_y: np.ndarray, train_y_resid: np.ndarray) -> np.ndarray:
        cod = np.array([r2_score(train_y, self.inputs.c_ypred["train_ypred"])])
        for fw_ypred in self.inputs.fw_ypred:
            cod = np.concatenate((cod, [r2_score(train_y_resid, fw_ypred["train_ypred"])]))
        ranks = np.argsort(-cod)
        return ranks

    def _random_forest_cv(
            self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
            key: str) -> None:
        params = {
            "n_estimators": np.linspace(100, 1000, 4, dtype=int),
            "min_samples_split": np.linspace(0.01, 0.05, 5),
            "max_features": np.linspace(
                np.floor(train_x.shape[1] / 2), train_x.shape[1], train_x.shape[1], dtype=int)}
        rfr_cv = GridSearchCV(
            estimator=RandomForestRegressor(criterion="friedman_mse"), param_grid=params)
        rfr_cv.fit(train_x, train_y)
        test_ypred = rfr_cv.predict(test_x)

        self._results["results"][f"r_{key}"] = np.corrcoef(
            test_y.T, test_ypred[np.newaxis, :])[0, 1]
        self._results["results"][f"cod_{key}"] = rfr_cv.score(test_x, test_y)
        self._results["results"][f"ypred_{key}"] = test_ypred

    def _run_interface(self, runtime):
        key = f"repeat{self.inputs.repeat}_fold{self.inputs.fold}"
        key_out = f"integrated_{key}_level{self.inputs.config['level']}"
        features = np.array(["conf"] + self.inputs.features)

        all_sub = sum(self.inputs.sublists.values(), [])
        train_sub = self.inputs.cv_split[key]
        test_sub = [subject for subject in all_sub if subject not in train_sub]
        train_x, train_y = self._extract_data(train_sub, "train_ypred")
        test_x, test_y = self._extract_data(test_sub, "test_ypred")
        train_y_resid, _ = pheno_reg_conf(
            train_y, train_x[:, 0].reshape(-1, 1), test_y, test_x[:, 0].reshape(-1, 1))
        feature_ranks = self._train_ranks(train_y, train_y_resid)

        self._results["results"] = {}
        self._results["results"][f"rank_{key_out}"] = np.array(features)[feature_ranks]
        for n_feature in range(2, train_x.shape[1]+1):
            key_out = f"integrated_{key}_level{self.inputs.config['level']}_{n_feature}features"
            x_ind = feature_ranks[:n_feature]
            self._random_forest_cv(train_x[:, x_ind], train_y, test_x[:, x_ind], test_y, key_out)

        return runtime
