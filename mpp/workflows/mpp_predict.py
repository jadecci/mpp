from pathlib import Path
import argparse
import configparser
import logging

import nipype.pipeline as pe

from mpp.interfaces.data import InitFeatures, PredictionSave
from mpp.interfaces.crossval import (
    CrossValSplit, FeaturewiseModel, ConfoundsModel, IntegratedFeaturesModel)
from mpp.interfaces.features import CVFeatures

base_dir = Path(__file__).resolve().parent.parent
logging.getLogger('datalad').setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Multimodal psychometric prediction',
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
    parser.add_argument(
        '--datasets', nargs='+', dest='datasets', required=True,
        help='Datasets for cross-validation')
    parser.add_argument(
        '--target', type=str, dest='target', required=True,
        help='Phenotype to use as prediction target')
    parser.add_argument(
        '--levels', nargs='+', dest='levels', default=['4'], help='Parcellation levels.')
    parser.add_argument(
        '--features', nargs='+', dest='features', help='Absolute paths to extracted features')
    parser.add_argument(
        '--phenotypes', nargs='+', dest='phenotypes',
        help='Absolute paths to phenotype directories')
    parser.add_argument(
        '--config', type=Path, dest='config', default=Path(base_dir, 'data', 'default.config'),
        help='Custom configuration file')
    parser.add_argument(
        '--ext_data', nargs='+', dest='ext_data', default=None,
        help='Dataset(s) as external test set')
    parser.add_argument(
        '--ext_features', nargs='+', dest='ext_features', default=None,
        help='Absolute path to extracted features of each external test set.')
    parser.add_argument(
        '--work_dir', type=Path, dest='work_dir', default=Path.cwd(), help='Work directory')
    parser.add_argument(
        '--output_dir', type=Path, dest='output_dir', default=Path.cwd(), help='output directory')
    parser.add_argument(
        '--overwrite', dest='overwrite', action="store_true", help='overwrite existing results')
    parser.add_argument(
        '--condordag', dest='condordag', action='store_true', help='submit graph to HTCondor')
    parser.add_argument(
        '--wrapper', type=str, dest='wrapper', default='', help='wrapper script for HTCondor')
    parser.add_argument(
        '--model', type=str, dest='model', default='', help='run a certain model')
    args = parser.parse_args()

    # Configuration file
    config_parse = configparser.ConfigParser()
    config_parse.read(args.config)
    config = {option: config_parse['USER'][option] for option in config_parse['USER']}

    # Preparations
    features_dir = dict(zip(args.datasets, args.features))
    mp_wf = pe.Workflow('mp_wf', base_dir=args.work_dir)
    init_data = pe.Node(
        InitFeatures(features_dir=features_dir,
                     phenotypes_dir=dict(zip(args.datasets, args.phenotypes)),
                     phenotype=args.target),
        name='init_data')
    cv_split = pe.Node(CrossValSplit(config=config), name='cv_split')
    mp_wf.connect([(init_data, cv_split, [('sublists', 'sublists')])])

    # Features to estimate during cross-validation
    features_iterables = [
        ('repeat', list(range(int(config['n_repeats'])))),
        ('fold', list(range(int(config['n_folds'])))),
        ('level', args.levels)]
    features = pe.Node(
        CVFeatures(config=config, features_dir=features_dir),
        name='features', iterables=features_iterables)
    mp_wf.connect([
        (init_data, features, [('sublists', 'sublists')]),
        (cv_split, features, [('cv_split', 'cv_split')])])

    # Feature-wise models
    fw_model = pe.Node(FeaturewiseModel(config=config, features_dir=features_dir), name='fw_model')
    fw_save = pe.JoinNode(
        PredictionSave(
            output_dir=args.output_dir, overwrite=args.overwrite, phenotype=args.target,
            type='featurewise'),
        name='fw_save', joinfield=['results'], joinsource='features')
    mp_wf.connect([
        (init_data, fw_model, [
            ('sublists', 'sublists'), ('confounds', 'confounds'), ('phenotypes', 'phenotypes')]),
        (cv_split, fw_model, [('cv_split', 'cv_split')]),
        (features, fw_model, [
            ('embeddings', 'embeddings'), ('params', 'params'), ('level', 'level'),
            ('repeat', 'repeat'), ('fold', 'fold')]),
        (fw_model, fw_save, [('results', 'results')])])

    # Confound models
    conf_model = pe.Node(ConfoundsModel(config=config), name='conf_model')
    conf_save = pe.JoinNode(
        PredictionSave(
            output_dir=args.output_dir, overwrite=args.overwrite, phenotype=args.target,
            type='confounds'),
        name='conf_save', joinfield=['results'], joinsource='features')
    mp_wf.connect([
        (init_data, conf_model, [
            ('sublists', 'sublists'), ('confounds', 'confounds'), ('phenotypes', 'phenotypes')]),
        (features, conf_model, [('repeat', 'repeat'), ('fold', 'fold')]),
        (cv_split, conf_model, [('cv_split', 'cv_split')]),
        (conf_model, conf_save, [('results', 'results')])])

    # Integrated-features set models
    conf_model = pe.Node(ConfoundsModel(config=config), name='conf_model')
    conf_save = pe.JoinNode(
        PredictionSave(
            output_dir=args.output_dir, overwrite=args.overwrite, phenotype=args.target,
            type='confounds'),
        name='conf_save', joinfield=['results'], joinsource='features')
    mp_wf.connect([
        (init_data, conf_model, [
            ('sublists', 'sublists'), ('confounds', 'confounds'), ('phenotypes', 'phenotypes')]),
        (features, conf_model, [('repeat', 'repeat'), ('fold', 'fold')]),
        (cv_split, conf_model, [('cv_split', 'cv_split')]),
        (conf_model, conf_save, [('results', 'results')])])

    mp_wf.config['execution']['try_hard_link_datasink'] = 'false'
    mp_wf.config['execution']['crashfile_format'] = 'txt'
    mp_wf.config['execution']['stop_on_first_crash'] = 'true'
    mp_wf.config['monitoring']['enabled'] = 'true'

    mp_wf.write_graph()
    if args.condordag:
        mp_wf.run(
            plugin='CondorDAGMan',
            plugin_args={
                'wrapper_cmd': args.wrapper,
                'dagman_args': f'-outfile_dir {args.work_dir} -import_env',
                'override_specs': 'request_memory = 5 GB\nrequest_cpus = 1'})
    else:
        mp_wf.run()


if __name__ == '__main__':
    main()
