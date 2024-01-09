from pathlib import Path
import argparse
import configparser

import nipype.pipeline as pe

from mpp.interfaces.crossval import (
    CrossValSplit, FeaturewiseModel, ConfoundsModel, IntegratedFeaturesModel)
from mpp.interfaces.data import InitFeatures, PredictionSave
from mpp.interfaces.features import CVFeatures

base_dir = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal brain-based psychometric prediction",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--target", type=str, dest="target", required=True, help="Prediction target")
    required.add_argument(
        "--features_dir", type=Path, dest="features_dir", required=True,
        help="Absolute paths to extracted features")
    required.add_argument(
        '--pheno_dir', nargs='+', dest='pheno_dir', required=True,
        help='Absolute paths to phenotype directories')
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "--levels", nargs="+", dest="levels", default=["4"], help="Parcellation levels.")
    optional.add_argument(
        "--config", type=Path, dest="config", default=Path(base_dir, "default.config"),
        help="Configuration file for cross-validation")
    optional.add_argument(
        "--work_dir", type=Path, dest="work_dir", default=Path.cwd(), help="Work directory")
    optional.add_argument(
        "--output_dir", type=Path, dest="output_dir", default=Path.cwd(), help="output directory")
    optional.add_argument(
        "--overwrite", dest="overwrite", action="store_true", help="overwrite existing results")
    optional.add_argument(
        "--condordag", dest="condordag", action="store_true", help="submit graph to HTCondor")
    config = vars(parser.parse_args())

    # Configuration file
    config_parse = configparser.ConfigParser()
    config_parse.read(config["config"])
    config.update({option: config_parse["USER"][option] for option in config_parse["USER"]})

    # Preparations
    mpp_wf = pe.Workflow("mpp_wf", base_dir=config["work_dir"])
    init_data = pe.Node(InitFeatures(config=config), "init_data")
    cv_split = pe.Node(CrossValSplit(config=config), name='cv_split')
    mpp_wf.connect([(init_data, cv_split, [('sublists', 'sublists')])])

    # Features to estimate during cross-validation
    features_iterables = [
        ('repeat', list(range(int(config['n_repeats'])))),
        ('fold', list(range(int(config['n_folds'])))),
        ('level', args.levels)]
    features = pe.Node(
        CVFeatures(config=config, features_dir=features_dir),
        name='features', iterables=features_iterables)
    mpp_wf.connect([
        (init_data, features, [('sublists', 'sublists')]),
        (cv_split, features, [('cv_split', 'cv_split')])])

    # Feature-wise models
    fw_model = pe.Node(FeaturewiseModel(config=config, features_dir=features_dir), name='fw_model')
    fw_save = pe.JoinNode(
        PredictionSave(
            output_dir=args.output_dir, overwrite=args.overwrite, phenotype=args.target,
            type='featurewise'),
        name='fw_save', joinfield=['results'], joinsource='features')
    mpp_wf.connect([
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
    mpp_wf.connect([
        (init_data, conf_model, [
            ('sublists', 'sublists'), ('confounds', 'confounds'), ('phenotypes', 'phenotypes')]),
        (features, conf_model, [('repeat', 'repeat'), ('fold', 'fold')]),
        (cv_split, conf_model, [('cv_split', 'cv_split')]),
        (conf_model, conf_save, [('results', 'results')])])

    # Integrated-features set models
    if_model = pe.Node(IntegratedFeaturesModel(config=config), name='if_model')
    if_save = pe.JoinNode(
        PredictionSave(
            output_dir=args.output_dir, overwrite=args.overwrite, phenotype=args.target,
            type='integratedfeatures'),
        name='if_save', joinfield=['results'], joinsource='features')
    mpp_wf.connect([
        (init_data, if_model, [('sublists', 'sublists'),('phenotypes', 'phenotypes')]),
        (cv_split, if_model, [('cv_split', 'cv_split')]),
        (features, if_model, [('level', 'level'), ('repeat', 'repeat'), ('fold', 'fold')]),
        (fw_model, if_model, [('fw_ypred', 'fw_ypred')]),
        (conf_model, if_model, [('c_ypred', 'c_ypred')]),
        (if_model, if_save, [('results', 'results')])])

    mpp_wf.config['execution']['try_hard_link_datasink'] = 'false'
    mpp_wf.config['execution']['crashfile_format'] = 'txt'
    mpp_wf.config['execution']['stop_on_first_crash'] = 'true'
    mpp_wf.config['monitoring']['enabled'] = 'true'

    mpp_wf.write_graph()
    if args.condordag:
        mpp_wf.run(
            plugin='CondorDAGMan',
            plugin_args={
                'wrapper_cmd': args.wrapper,
                'dagman_args': f'-outfile_dir {args.work_dir} -import_env',
                'override_specs': 'request_memory = 10 GB\nrequest_cpus = 1'})
    else:
        mpp_wf.run()


if __name__ == '__main__':
    main()
