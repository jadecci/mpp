from pathlib import Path
import argparse
import configparser
import logging

import nipype.pipeline as pe

from mpp.interfaces.data import InitFeatures, PredictionSave
from mpp.interfaces.crossval import (
    CrossValSplit, RegionwiseModel, FeaturewiseModel, ConfoundsModel, IntegratedFeaturesModel,
    RandomPatchesModel)
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
        '--levels', nargs='+', dest='levels', default=['1', '2', '3', '4'],
        help='parcellation levels')
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
        '--only', type=str, dest='only', default='', help='only run a certain model')
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

    # Models
    if args.only == 'regionwise':
        mp_wf = regionwise_wf(mp_wf, init_data, cv_split, features, config, features_dir, args)
    elif args.only == 'confound':
        mp_wf, _ = confound_wf(mp_wf, init_data, cv_split, args)
    elif args.only == 'randompatches':
        mp_wf = randompatches_wf(mp_wf, init_data, cv_split, features, args)
    else:
        mp_wf, fw_model = featurewise_wf(
            mp_wf, init_data, cv_split, features, config, features_dir, args)
        if args.only != 'featurewise':
            mp_wf, conf_model = confound_wf(mp_wf, init_data, cv_split, args)
            mp_wf = integratedfeatures_wf(
                mp_wf, init_data, cv_split, features, fw_model, conf_model, args)

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
                'override_specs': 'requirements=(Machine!="cpu43.htc.inm7.de")'}) # do not use cpu43 for now
    else:
        mp_wf.run()


def regionwise_wf(
        wf: pe.Workflow, init_data: pe.Node, cv_split: pe.Node, features: pe.Node, config: dict,
        features_dir: dict, args) -> pe.Workflow:
    rw_model = pe.Node(
        RegionwiseModel(mode='test', config=config, features_dir=features_dir), name='rw_test')
    rw_save = pe.JoinNode(
        PredictionSave(
            output_dir=args.output_dir, overwrite=args.overwrite, phenotype=args.target,
            type='regionwise'),
        name='rw_save', joinfield=['results'], joinsource='features')

    wf.connect([
        (init_data, rw_model, [('sublists', 'sublists'), ('phenotypes', 'phenotypes')]),
        (cv_split, rw_model, [('cv_split', 'cv_split')]),
        (features, rw_model, [
            ('embeddings', 'embeddings'), ('params', 'params'), ('level', 'level'),
            ('repeat', 'repeat'), ('fold', 'fold')]),
        (rw_model, rw_save, [('results', 'results')])])

    return wf


def featurewise_wf(
        wf: pe.Workflow, init_data: pe.Node, cv_split: pe.Node, features: pe.Node, config: dict,
        features_dir: dict, args) -> tuple[pe.Workflow, pe.Node]:
    fw_model = pe.Node(FeaturewiseModel(config=config, features_dir=features_dir), name='fw_model')
    fw_save = pe.JoinNode(
        PredictionSave(
            output_dir=args.output_dir, overwrite=args.overwrite, phenotype=args.target,
            type='featurewise'),
        name='fw_save', joinfield=['results'], joinsource='features')

    wf.connect([
        (init_data, fw_model, [('sublists', 'sublists'), ('phenotypes', 'phenotypes')]),
        (cv_split, fw_model, [('cv_split', 'cv_split')]),
        (features, fw_model, [
            ('embeddings', 'embeddings'), ('params', 'params'), ('level', 'level'),
            ('repeat', 'repeat'), ('fold', 'fold')]),
        (fw_model, fw_save, [('results', 'results')])])

    return wf, fw_model


def confound_wf(
        wf: pe.Workflow, init_data: pe.Node, cv_split:
        pe.Node, args) -> tuple[pe.Workflow, pe.Node]:
    conf_model = pe.Node(ConfoundsModel(), name='conf_model')
    conf_save = pe.JoinNode(
        PredictionSave(
            output_dir=args.output_dir, overwrite=args.overwrite, phenotype=args.target,
            type='confounds'),
        name='conf_save', joinfield=['results'], joinsource='features')

    wf.connect([
        (init_data, conf_model, [
            ('sublists', 'sublists'), ('confounds', 'confounds'), ('phenotypes', 'phenotypes'),
            ('level', 'level'), ('repeat', 'repeat'), ('fold', 'fold')]),
        (cv_split, conf_model, [('cv_split', 'cv_split')]),
        (conf_model, conf_save, [('results', 'results')])])

    return wf, conf_model


def integratedfeatures_wf(
        wf: pe.Workflow, init_data: pe.Node, cv_split: pe.Node, features: pe.Node,
        fw_model: pe.Node, conf_model: pe.Node, args) -> pe.Workflow:
    if_model = pe.Node(IntegratedFeaturesModel(), name='if_model')
    if_save = pe.JoinNode(
        PredictionSave(
            output_dir=args.output_dir, overwrite=args.overwrite, phenotype=args.target,
            type='integratedfeatures'),
        name='if_save', joinfield=['results'], joinsource='features')

    wf.connect([
        (init_data, if_model, [('sublists', 'sublists'),('phenotypes', 'phenotypes')]),
        (cv_split, if_model, [('cv_split', 'cv_split')]),
        (features, if_model, [('level', 'level'), ('repeat', 'repeat'), ('fold', 'fold')]),
        (fw_model, if_model, [('fw_ypred', 'fw_ypred')]),
        (conf_model, if_model, [('c_ypred', 'c_ypred')]),
        (if_model, if_save, [('results', 'results')])])

    return wf


def randompatches_wf(
            wf: pe.Workflow, init_data: pe.Node, cv_split: pe.Node, features: pe.Node,
            args) -> pe.Workflow:
    rp_model = pe.Node(RandomPatchesModel(), name='rp_model')
    rp_save = pe.JoinNode(
        PredictionSave(
            output_dir=args.output_dir, overwrite=args.overwrite, phenotype=args.target,
            type='randompatches'),
        name='rp_save', joinfield=['results'], joinsource='features')

    wf.connect([
        (init_data, rp_model, [('sublists', 'sublists'), ('phenotypes', 'phenotypes')]),
        (cv_split, rp_model, [('cv_split', 'cv_split')]),
        (features, rp_model, [('level', 'level'), ('repeat', 'repeat'), ('fold', 'fold')]),
        (rp_model, rp_save, [('results', 'results')])])

    return wf


if __name__ == '__main__':
    main()
