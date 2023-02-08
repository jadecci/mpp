import pandas as pd
from os import getcwd, path
import argparse, configparser
import logging

import nipype.pipeline as pe
from nipype.interfaces import utility as niu

from mpp.interfaces.data import InitFeatures
from mpp.interfaces.crossval import CrossValSplit

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
    parser.add_argument('--workdir', type=str, dest='work_dir', default=getcwd(), help='Work directory')
    parser.add_argument('--output_dir', type=str, dest='output_dir', default=getcwd(), help='output directory')
    parser.add_argument('--overwrite', dest='overwrite', action="store_true", help='overwrite existing results')
    parser.add_argument('--condordag', dest='condordag', action='store_true', help='submit graph workflow to HTCondor')
    parser.add_argument('--wrapper', type=str, dest='wrapper', default='', help='wrapper script for HTCondor')
    args = parser.parse_args()

    ## Configuration file
    config_parse = configparser.ConfigParser()
    config_parse.read(args.config)
    config = {option: config_parse['USER'][option] for option in config_parse['USER']}

    ## Workflow nodes
    mp_wf = pe.Workflow('mp_wf', base_dir=args.work_dir)
    init_data = pe.Node(InitFeatures(features_dir=dict(zip(args.datasets, args.features)), phenotype=args.target),
                        name='init_data')
    cv_split = pe.Node(CrossValSplit(config=config), name='cv_split')

    ## Workflow connections
    mp_wf.connect([(init_data, cv_split, [('sublists', 'sublists')])])

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