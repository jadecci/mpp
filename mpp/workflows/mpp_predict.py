import pandas as pd
from os import getcwd, path
import itertools
import argparse, configparser
import logging

import nipype.pipeline as pe
from nipype.interfaces import utility as niu

from mpp.interfaces.data import InitFeatures, RegionwiseSave
from mpp.interfaces.crossval import CrossValSplit, RegionwiseModel, FeatureSelect
from mpp.interfaces.features import Gradient, AC

base_dir = path.join(path.dirname(path.realpath(__file__)), '..')
logging.getLogger('datalad').setLevel(logging.WARNING)

def main():
    parser = argparse.ArgumentParser(description='Multimodal psychometric prediction',
                formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
    parser.add_argument('datasets', nargs='+', help='Datasets for cross-validation')
    parser.add_argument('features', nargs='+', help='Absolute paths to extracted features of each dataset.')
    parser.add_argument('phenotypes', nargs='+', help='Absolute paths to phenotype directory of each dataset')
    parser.add_argument('target', type=str, help='Phenotype to use as prediction target')
    parser.add_argument('--config', type=str, dest='config', default=path.join(base_dir, 'data', 'default.config'), 
                        help='Custom configuration file')
    parser.add_argument('--ext_data', nargs='+', dest='ext_data', default=None, help='Dataset(s) as external test set')
    parser.add_argument('--ext_features', nargs='+', dest='ext_features', default=None,
                        help='Absolute path to extracted features of each external test set.')
    parser.add_argument('--work_dir', type=str, dest='work_dir', default=getcwd(), help='Work directory')
    parser.add_argument('--output_dir', type=str, dest='output_dir', default=getcwd(), help='output directory')
    parser.add_argument('--overwrite', dest='overwrite', action="store_true", help='overwrite existing results')
    parser.add_argument('--condordag', dest='condordag', action='store_true', help='submit graph workflow to HTCondor')
    parser.add_argument('--wrapper', type=str, dest='wrapper', default='', help='wrapper script for HTCondor')
    args = parser.parse_args()

    ## Configuration file
    config_parse = configparser.ConfigParser()
    config_parse.read(args.config)
    config = {option: config_parse['USER'][option] for option in config_parse['USER']}

    ## Preparations
    mp_wf = pe.Workflow('mp_wf', base_dir=args.work_dir)
    init_data = pe.Node(InitFeatures(features_dir=dict(zip(args.datasets, args.features)), 
                                     phenotypes_dir=dict(zip(args.datasets, args.phenotypes)), 
                                     phenotype=args.target),
                        name='init_data')
    cv_split = pe.Node(CrossValSplit(config=config), name='cv_split')
    gradient = pe.Node(Gradient(config=config), name='gradient')
    ac = pe.Node(AC(config=config), name='ac')
    cv_split_perm = pe.Node(CrossValSplit(config=config, permutation=True), name='cv_split_perm')
    gradient_perm = pe.Node(Gradient(config=config), name='gradient_perm')
    ac_perm = pe.Node(AC(config=config), name='ac_perm')

    mp_wf.connect([(init_data, cv_split, [('sublists', 'sublists')]),
                   (init_data, gradient, [('sublists', 'sublists'),
                                          ('image_features', 'image_features')]),
                   (init_data, ac, [('sublists', 'sublists'),
                                    ('image_features', 'image_features')]),
                   (init_data, cv_split_perm, [('sublists', 'sublists')]),
                   (init_data, gradient_perm, [('sublists', 'sublists'),
                                               ('image_features', 'image_features')]),
                   (init_data, ac_perm, [('sublists', 'sublists'),
                                         ('image_features', 'image_features')]),
                   (cv_split, gradient, [('cv_split', 'cv_split')]),
                   (cv_split, ac, [('cv_split', 'cv_split')]),
                   (cv_split_perm, gradient_perm, [('cv_split', 'cv_split')]),
                   (cv_split_perm, ac_perm, [('cv_split', 'cv_split')])])

    ## Region-wise models
    n_region = {'1': 116, '2': 232, '3': 350, '4': 454, 'conf': 1}
    levels = [[key] * n_region[key] for key in n_region]
    levels = list(itertools.chain.from_iterable(levels))
    regions = [[region for region in range(n_region[key])] for key in n_region]
    regions = list(itertools.chain.from_iterable(regions))

    rw_inputnode = pe.Node(niu.IdentityInterface(fields=['levels', 'regions'], levels=levels, regions=regions),
                        name='inputnode')
    rw_validate = pe.MapNode(RegionwiseModel(mode='validate', config=config), name='regionwise_validate', 
                                             iterfield=['level', 'region'])
    rw_select = pe.Node(FeatureSelect(), name='feature_select')
    rw_test = pe.MapNode(RegionwiseModel(mode='test', config=config), name='regionwise_test',
                                        iterfield=['level', 'region'])
    rw_save = pe.Node(RegionwiseSave(output_dir=args.output_dir, overwrite=args.overwrite), name='regionwise_save')
    
    mp_wf.connect([(init_data, rw_validate, [('sublists', 'sublists'),
                                             ('image_features', 'image_features'),
                                             ('confounds', 'confounds'),
                                             ('phenotypes', 'phenotypes'),
                                             ('phenotypes_perm', 'phenotypes_perm')]),
                   (init_data, rw_test, [('sublists', 'sublists'),
                                         ('image_features', 'image_features'),
                                         ('confounds', 'confounds'),
                                         ('phenotypes', 'phenotypes')]),
                   (cv_split, rw_validate, [('cv_split', 'cv_split')]),
                   (cv_split, rw_test, [('cv_split', 'cv_split')]),
                   (gradient, rw_validate, [('gradients', 'gradients')]),
                   (gradient, rw_test, [('gradients', 'gradients')]),
                   (ac, rw_validate, [('ac', 'ac')]),
                   (ac, rw_test, [('ac', 'ac')]),
                   (cv_split_perm, rw_validate, [('cv_split', 'cv_split_perm')]),
                   (gradient_perm, rw_validate, [('gradients', 'gradients_perm')]),
                   (ac_perm, rw_validate, [('ac', 'ac_perm')]),
                   (rw_inputnode, rw_validate, [('levels', 'level'),
                                                ('regions', 'region')]),
                   (rw_inputnode, rw_select, [('levels', 'levels'),
                                              ('regions', 'regions')]),
                   (rw_validate, rw_select, [('p_r', 'p_r'),
                                             ('p_cod', 'p_cod')]),
                   (rw_select, rw_test, [('selected_levels', 'level'),
                                         ('selected_regions', 'region')]),
                   (rw_select, rw_save, [('selected_levels', 'levels'),
                                         ('selected_regions', 'regions')]),
                   (rw_test, rw_save, [('r', 'r'),
                                       ('cod', 'cod'),
                                       ('selected', 'selected')])])

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

    return wf

if __name__ == '__main__':
    main()