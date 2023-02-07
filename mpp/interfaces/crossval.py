from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import itertools
import numpy as np
from os import path
import logging

from sklearn.model_selection import RepeatedStratifiedKFold

base_dir = path.join(path.dirname(path.realpath(__file__)), '..')
logging.getLogger('datalad').setLevel(logging.WARNING)

### CrossValSplit: generate cross-validation splits

class _CrossValSplitInputSpec(BaseInterfaceInputSpec):
    sublists = traits.Dict(dtype=list, desc='list of subjects available in each dataset')
    config = traits.Dict(desc='configuration settings')

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
        rskf = RepeatedStratifiedKFold(n_splits=self.inputs.config['n_splits'], 
                                       n_repeats=self.inputs.config['n_repeats'], 
                                       random_state=self.inputs.config['cv_seed'])
        for fold, (_, test_ind) in enumerate(rskf.split(subjects, datasets)):
            key = f'Repeat{int(np.floor(fold/10))}Fold{int(fold%10)}'
            self._results['cv_split'][key] = subjects[test_ind]

        return runtime