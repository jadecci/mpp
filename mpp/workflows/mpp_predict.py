from pathlib import Path
import argparse, configparser
import logging

import nipype.pipeline as pe
from nipype.interfaces import utility as niu

from mpp.interfaces.data import InitFeatures, RegionwiseSave
from mpp.interfaces.crossval import CrossValSplit, RegionwiseModel, RegionSelect
from mpp.interfaces.features import GradientAC

base_dir = Path(__file__).resolve().parent.parent
logging.getLogger('datalad').setLevel(logging.WARNING)

def main():
    parser = argparse.ArgumentParser(description='Multimodal psychometric prediction',
                formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
    parser.add_argument('datasets', nargs='+', help='Datasets for cross-validation')
    parser.add_argument('target', type=str, help='Phenotype to use as prediction target')
    parser.add_argument('--levels', nargs='+', dest='levels', default=['1', '2', '3', '4'], help='parcellation levels')
    parser.add_argument('--features', nargs='+', dest='features', help='Absolute paths to extracted features')
    parser.add_argument('--phenotypes', nargs='+', dest='phenotypes', help='Absolute paths to phenotype directories')
    parser.add_argument('--config', type=Path, dest='config', default=Path(base_dir, 'data', 'default.config'), 
                        help='Custom configuration file')
    parser.add_argument('--ext_data', nargs='+', dest='ext_data', default=None, help='Dataset(s) as external test set')
    parser.add_argument('--ext_features', nargs='+', dest='ext_features', default=None,
                        help='Absolute path to extracted features of each external test set.')
    parser.add_argument('--work_dir', type=Path, dest='work_dir', default=Path.cwd(), help='Work directory')
    parser.add_argument('--output_dir', type=Path, dest='output_dir', default=Path.cwd(), help='output directory')
    parser.add_argument('--overwrite', dest='overwrite', action="store_true", help='overwrite existing results')
    parser.add_argument('--condordag', dest='condordag', action='store_true', help='submit graph workflow to HTCondor')
    parser.add_argument('--wrapper', type=str, dest='wrapper', default='', help='wrapper script for HTCondor')
    args = parser.parse_args()

    ## Configuration file
    config_parse = configparser.ConfigParser()
    config_parse.read(args.config)
    config = {option: config_parse['USER'][option] for option in config_parse['USER']}

    ## Preparations
    features_dir = dict(zip(args.datasets, args.features))
    mp_wf = pe.Workflow('mp_wf', base_dir=args.work_dir)
    init_data = pe.Node(InitFeatures(features_dir=features_dir, 
                                     phenotypes_dir=dict(zip(args.datasets, args.phenotypes)), 
                                     phenotype=args.target),
                        name='init_data')
    cv_split = pe.Node(CrossValSplit(config=config), name='cv_split')
    cv_split_perm = pe.Node(CrossValSplit(config=config, permutation=True), name='cv_split_perm')

    mp_wf.connect([(init_data, cv_split, [('sublists', 'sublists')]),
                   (init_data, cv_split_perm, [('sublists', 'sublists')])])

    ## Features to estimate during cross-validation
    features_iterables = [('repeat', list(range(int(config['n_repeats'])))),
                          ('fold', list(range(int(config['n_folds'])))),
                          ('level', args.levels)]
    features = pe.Node(GradientAC(config=config, features_dir=features_dir), 
                       name='features', iterables=features_iterables, mem_gb=5)

    mp_wf.connect([(init_data, features, [('sublists', 'sublists')]),
                   (cv_split, features, [('cv_split', 'cv_split')]),
                   (cv_split_perm, features, [('cv_split', 'cv_split_perm')])])

    ## Region-wise models
    rw_validate = pe.Node(RegionwiseModel(mode='validate', config=config, features_dir=features_dir),
                          name='rw_validate', mem_gb=5)
    rw_select = pe.JoinNode(RegionSelect(), name='rw_select', joinfield=['results'], joinsource='features', mem_gb=5)
    rw_select.inputs.levels = args.levels
    rw_select.inputs.config = config
    rw_test = pe.Node(RegionwiseModel(mode='test', config=config, features_dir=features_dir), 
                      name='rw_test', mem_gb=5)
    rw_save = pe.JoinNode(RegionwiseSave(output_dir=args.output_dir, overwrite=args.overwrite), name='rw_save',
                          joinfield=['results', 'selected_features'], joinsource='features', synchronize=True, mem_gb=5)
    
    mp_wf.connect([(init_data, rw_validate, [('sublists', 'sublists'),
                                             ('confounds', 'confounds'),
                                             ('phenotypes', 'phenotypes'),
                                             ('phenotypes_perm', 'phenotypes_perm')]),
                   (cv_split, rw_validate, [('cv_split', 'cv_split')]),
                   (cv_split_perm, rw_validate, [('cv_split', 'cv_split_perm')]),
                   (features, rw_validate, [('embeddings', 'embeddings'),
                                            ('params', 'params'),
                                            ('level', 'level'),
                                            ('repeat', 'repeat'),
                                            ('fold', 'fold')]),
                   (rw_validate, rw_select, [('results', 'results')]),
                   (init_data, rw_test, [('sublists', 'sublists'),
                                         ('confounds', 'confounds'),
                                         ('phenotypes', 'phenotypes')]),
                   (cv_split, rw_test, [('cv_split', 'cv_split')]),
                   (features, rw_test, [('embeddings', 'embeddings'),
                                        ('params', 'params'),
                                        ('level', 'level'),
                                        ('repeat', 'repeat'),
                                        ('fold', 'fold')]),
                   (rw_select, rw_test, [('selected', 'selected')]),
                   (rw_select, rw_save, [('selected', 'selected_regions')]),
                   (rw_test, rw_save, [('results', 'results'),
                                       ('selected', 'selected_features')])])

    ## config
    mp_wf.config['execution']['try_hard_link_datasink'] = 'false'
    mp_wf.config['execution']['crashfile_format'] = 'txt'
    mp_wf.config['execution']['stop_on_first_crash'] = 'true'
    mp_wf.config['monitoring']['enabled'] = 'true'

    mp_wf.write_graph()
    if args.condordag:
        mp_wf.run(plugin='CondorDAGMan', plugin_args={'dagman_args': f'-outfile_dir {args.work_dir}',
                                                      'wrapper_cmd': args.wrapper})
    else:
        mp_wf.run()

if __name__ == '__main__':
    main()