from pathlib import Path
import argparse
import configparser
import logging

import nipype.pipeline as pe

from mpp.interfaces.data import InitFeatures, PredictionSave
from mpp.interfaces.crossval import (
    CrossValSplit, RegionwiseModel, ModalitywiseModel, FeaturewiseModel, IntegratedFeaturesModel)
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
        '--condordag', dest='condordag', action='store_true',
        help='submit graph workflow to HTCondor')
    parser.add_argument(
        '--wrapper', type=str, dest='wrapper', default='', help='wrapper script for HTCondor')
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
    # cv_split_perm = pe.Node(CrossValSplit(config=config, permutation=True), name='cv_split_perm')
    mp_wf.connect([(init_data, cv_split, [('sublists', 'sublists')]),])
    #   (init_data, cv_split_perm, [('sublists', 'sublists')])])

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
        (cv_split, features, [('cv_split', 'cv_split')]),])
        # (cv_split_perm, features, [('cv_split', 'cv_split_perm')])])

    # Region-wise models
    #rw_validate = pe.Node(
    #    RegionwiseModel(mode='validate', config=config, features_dir=features_dir),
    #    name='rw_validate')
    #rw_select = pe.JoinNode(
    #    RegionSelect(
    #        levels=args.levels, output_dir=args.output_dir, overwrite=args.overwrite, config=config,
    #        phenotype=args.target),
    #    name='rw_select', joinfield=['results'], joinsource='features')
    #rw_test = pe.Node(
    #    RegionwiseModel(mode='test', config=config, features_dir=features_dir), name='rw_test')
    #rw_save = pe.JoinNode(
    #    PredictionSave(
    #        output_dir=args.output_dir, overwrite=args.overwrite, phenotype=args.target,
    #        type='regionwise'),
    #    name='rw_save', joinfield=['results'], joinsource='features')
    #mp_wf.connect([
    #    (init_data, rw_validate, [
    #        ('sublists', 'sublists'), ('phenotypes', 'phenotypes'),
    #        ('phenotypes_perm', 'phenotypes_perm')]),
    #    (cv_split, rw_validate, [('cv_split', 'cv_split')]),
    #    (cv_split_perm, rw_validate, [('cv_split', 'cv_split_perm')]),
    #    (features, rw_validate, [
    #        ('embeddings', 'embeddings'), ('params', 'params'), ('level', 'level'),
    #        ('repeat', 'repeat'), ('fold', 'fold')]),
    #    (rw_validate, rw_select, [('results', 'results')]),
    #    (init_data, rw_test, [('sublists', 'sublists'), ('phenotypes', 'phenotypes')]),
    #    (cv_split, rw_test, [('cv_split', 'cv_split')]),
    #    (features, rw_test, [
    #        ('embeddings', 'embeddings'), ('params', 'params'), ('level', 'level'),
    #        ('repeat', 'repeat'), ('fold', 'fold')]),
    #    (rw_select, rw_test, [('selected', 'selected')]),
    #    (rw_test, rw_save, [('results', 'results')])])

    # Modality-wise models
    mw_model = pe.Node(ModalitywiseModel(config=config, features_dir=features_dir), name='mw_model')
    mw_save = pe.JoinNode(
        PredictionSave(
            output_dir=args.output_dir, overwrite=args.overwrite, phenotype=args.target,
            type='modalitywise'),
        name='mw_save', joinfield=['results'], joinsource='features')
    mp_wf.connect([
        (init_data, mw_model, [('sublists', 'sublists'), ('phenotypes', 'phenotypes')]),
        (cv_split, mw_model, [('cv_split', 'cv_split')]),
        (features, mw_model, [
            ('embeddings', 'embeddings'), ('params', 'params'), ('level', 'level'),
            ('repeat', 'repeat'), ('fold', 'fold')]),
        (mw_model, mw_save, [('results', 'results')])])

    # Feature-wise models
    fw_model = pe.Node(FeaturewiseModel(config=config, features_dir=features_dir), name='fw_model')
    fw_save = pe.JoinNode(
        PredictionSave(
            output_dir=args.output_dir, overwrite=args.overwrite, phenotype=args.target,
            type='featurewise'),
        name='fw_save', joinfield=['results'], joinsource='features')
    mp_wf.connect([
        (init_data, fw_model, [('sublists', 'sublists'), ('phenotypes', 'phenotypes')]),
        (cv_split, fw_model, [('cv_split', 'cv_split')]),
        (features, fw_model, [
            ('embeddings', 'embeddings'), ('params', 'params'), ('level', 'level'),
            ('repeat', 'repeat'), ('fold', 'fold')]),
        (fw_model, fw_save, [('results', 'results')])])

    # Integrated features model
    if_model = pe.Node(IntegratedFeaturesModel(config=config), name='if_model')
    if_save = pe.JoinNode(
        PredictionSave(
            output_dir=args.output_dir, overwrite=args.overwrite, phenotype=args.target,
            type='integratedfeatures'),
        name='if_save', joinfield=['results'], joinsource='features')

    mp_wf.connect([
        (init_data, if_model, [('sublists', 'sublists'),('phenotypes', 'phenotypes')]),
        (cv_split, if_model, [('cv_split', 'cv_split')]),
        (features, if_model, [
    #        ('embeddings', 'embeddings'), ('params', 'params'),
            ('level', 'level'), ('repeat', 'repeat'), ('fold', 'fold')]),
    #    (rw_select, if_model, [('selected', 'selected_regions')]),
    #    (rw_test, if_model, [('rw_ypred', 'rw_ypred')]),
        (mw_model, if_model, [('mw_ypred', 'mw_ypred')]),
        (fw_model, if_model, [('fw_ypred', 'fw_ypred')]),
        (if_model, if_save, [('results', 'results')])])

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


if __name__ == '__main__':
    main()
