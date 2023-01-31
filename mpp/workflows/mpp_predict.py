import pandas as pd
from os import getcwd, path
import argparse
import logging

import nipype.pipeline as pe
from nipype.interfaces import utility as niu

from mpp.interfaces.data import InitFeatures

base_dir = path.join(path.dirname(path.realpath(__file__)), '..', '..')
logging.getLogger('datalad').setLevel(logging.WARNING)

def main():
    parser = argparse.ArgumentParser(description='Multimodal psychometric prediction',
                formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
    parser.add_argument('datasets', nargs='+', help='Datasets for cross-validation')
    parser.add_argument('features', nargs='+', help='Absolute path to extracted features of each dataset.')
    parser.add_argument('--ext_data', nargs='+', dest='ext_data', default=None, help='Dataset(s) as external test set')
    parser.add_argument('--ext_features', nargs='+', dest='ext_features', default=None,
                        help='Absolute path to extracted features of each external test set.')
    parser.add_argument('--workdir', type=str, dest='work_dir', default=getcwd(), help='Work directory')
    parser.add_argument('--output_dir', type=str, dest='output_dir', default=getcwd(), help='output directory')
    parser.add_argument('--overwrite', dest='overwrite', action="store_true", help='overwrite existing results')
    parser.add_argument('--condordag', dest='condordag', action='store_true', help='submit graph workflow to HTCondor')
    parser.add_argument('--wrapper', type=str, dest='wrapper', default='', help='wrapper script for HTCondor')
    args = parser.parse_args()

    ## 

    ## Overall workflow
    mp_wf = pe.Workflow('mp_wf', base_dir=args.work_dir)
    init_data = pe.Node(InitFeatures(features_dir=dict(zip(args.datasets, args.features))), name='init_data')

    ## config
    mp_wf.config['execution']['try_hard_link_datasink'] = 'false'
    mp_wf.config['execution']['crashfile_format'] = 'txt'
    mp_wf.config['execution']['stop_on_first_crash'] = 'true'
    mp_wf.config['monitoring']['enabled'] = 'true'

    ## Individual subject's workflow
    #sublist = pd.read_csv(args.sublist, header=None, squeeze=True)
    #for subject in sublist:
    #    subject_wf = init_subject_wf(args.dataset, subject, args.output_dir, args.overwrite)
    #    mf_wf.connect(init_data, 'dataset_dir', subject_wf, 'inputnode.dataset_dir')

    mp_wf.write_graph()
    if args.condordag:
        mp_wf.run(plugin='CondorDAGMan', plugin_args={'dagman_args': f'-outfile_dir {args.work_dir}',
                                                      'wrapper_cmd': args.wrapper})
    else:
        mp_wf.run()

if __name__ == '__main__':
    main()