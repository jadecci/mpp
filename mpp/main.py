from importlib.resources import files
from pathlib import Path
import argparse
import configparser

import nipype.pipeline as pe

from mpp.interfaces.crossval import (
    CrossValSplit, FeaturewiseModel, ConfoundsModel, IntegratedFeaturesModel)
from mpp.interfaces.data import PredictSublist, PredictionCombine, PredictionSave
from mpp.interfaces.features import CVFeatures
from mpp.utilities import feature_list
import mpp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal brain-based psychometric prediction",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
    required = parser.add_argument_group("required arguments")
    required.add_argument("--targets", nargs="+", dest="targets", required=True, help="Targets")
    required.add_argument("--datasets", nargs="+", dest="datasets", required=True, help="Datasets")
    required.add_argument(
        "--features_dir", type=Path, dest="features_dir", required=True,
        help="Absolute paths to extracted features")
    required.add_argument(
        "--sublists", nargs="+", dest="sublists", required=True,
        help="Subject lists (one list for each dataset, in the same order as --datasets)")
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "--level", type=str, dest="level", default="4", help="Parcellation level")
    optional.add_argument(
        "--config", type=Path, dest="config", default=files(mpp)/"default.config",
        help="Configuration file for cross-validation")
    optional.add_argument(
        "--hcpya_res", type=Path, desc="hcpya_res", default="", help="HCP-YA restricted data csv")
    optional.add_argument(
        "--work_dir", type=Path, dest="work_dir", default=Path.cwd(), help="Work directory")
    optional.add_argument(
        "--output_dir", type=Path, dest="output_dir", default=Path.cwd(), help="output directory")
    optional.add_argument(
        "--overwrite", dest="overwrite", action="store_true", help="overwrite existing results")
    optional.add_argument(
        "--condordag", dest="condordag", action="store_true", help="submit graph to HTCondor")
    config = vars(parser.parse_args())

    # Set-up
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    config_parse = configparser.ConfigParser()
    config_parse.read(config["config"])
    config.update({option: config_parse["USER"][option] for option in config_parse["USER"]})
    mpp_wf = pe.Workflow("mpp_wf", base_dir=config["work_dir"])
    mpp_wf.config["execution"]["try_hard_link_datasink"] = "false"
    mpp_wf.config["execution"]["crashfile_format"] = "txt"
    mpp_wf.config["execution"]["stop_on_first_crash"] = "true"

    # Preparations
    sublist = pe.Node(
        PredictSublist(config=config), "sublist", iterables=[("target", config["targets"])])
    cv_split = pe.Node(CrossValSplit(config=config), "cv_split")
    cv_iterables = [
        ("repeat", list(range(int(config["n_repeats"])))),
        ("fold", list(range(int(config["n_folds"]))))]
    features = pe.Node(CVFeatures(config=config), "features", iterables=cv_iterables)
    mpp_wf.connect([
        (sublist, cv_split, [("sublists", "sublists")]),
        (sublist, features, [("sublists", "sublists"), ("target", "target")]),
        (cv_split, features, [("cv_split", "cv_split")])])

    # Feature-wise models
    features_iterables = [("feature_type", feature_list(config["datasets"]))]
    fw_model = pe.Node(FeaturewiseModel(config=config), "fw_model", iterables=features_iterables)
    fw_combine = pe.JoinNode(
        PredictionCombine(config=config), "fw_combine",
        joinsource="fw_model", joinfield=["results"])
    fw_save = pe.JoinNode(
        PredictionSave(config=config, model_type="featurewise"), "fw_save",
        joinsource="features", joinfield=["results"], )
    mpp_wf.connect([
        (sublist, fw_model, [("sublists", "sublists"), ("target", "target")]),
        (cv_split, fw_model, [("cv_split", "cv_split")]),
        (features, fw_model, [
            ("cv_features_file", "cv_features_file"), ("repeat", "repeat"), ("fold", "fold")]),
        (fw_model, fw_combine, [("results", "results")]),
        (sublist, fw_save, [("target", "target")]),
        (fw_combine, fw_save, [("results", "results")])])

    # Confound models
    conf_model = pe.Node(ConfoundsModel(config=config), "conf_model")
    conf_save = pe.JoinNode(
        PredictionSave(config=config, model_type="confounds"), "conf_save",
        joinsource="features", joinfield=["results"])
    mpp_wf.connect([
        (sublist, conf_model, [("sublists", "sublists"), ("target", "target")]),
        (features, conf_model, [("repeat", "repeat"), ("fold", "fold")]),
        (cv_split, conf_model, [("cv_split", "cv_split")]),
        (sublist, conf_save, [("target", "target")]),
        (conf_model, conf_save, [("results", "results")])])

    # Integrated-features set models
    if_model = pe.JoinNode(
        IntegratedFeaturesModel(config=config, features=feature_list(config["datasets"])),
        "if_model", joinsource="fw_model", joinfield=["fw_ypred"])
    if_save = pe.JoinNode(
        PredictionSave(config=config, model_type="integrated"), "if_save",
        joinsource="features", joinfield=["results"])
    mpp_wf.connect([
        (sublist, if_model, [("sublists", "sublists"), ("target", "target")]),
        (cv_split, if_model, [("cv_split", "cv_split")]),
        (features, if_model, [("repeat", "repeat"), ("fold", "fold")]),
        (fw_model, if_model, [("fw_ypred", "fw_ypred")]),
        (conf_model, if_model, [("c_ypred", "c_ypred")]),
        (sublist, if_save, [("target", "target")]),
        (if_model, if_save, [("results", "results")])])

    # Run workflow
    mpp_wf.write_graph()
    if config["condordag"]:
        mpp_wf.run(
            plugin="CondorDAGMan",
            plugin_args={
                "dagman_args": f"-outfile_dir {config['work_dir']} -import_env",
                "wrapper_cmd": files(mpp) / "venv_wrapper.sh",
                "override_specs": "request_memory = 10 GB\nrequest_cpus = 1"})
    else:
        mpp_wf.run()


if __name__ == '__main__':
    main()
