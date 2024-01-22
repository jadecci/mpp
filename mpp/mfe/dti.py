from importlib.resources import files
from pathlib import Path
import argparse
import configparser

import nipype.pipeline as pe
import pandas as pd
from nipype.interfaces import fsl

from mpp.mfe.interfaces.data import InitDTIData, SaveDTIFeatures
from mpp.mfe.interfaces.diffusion import TBSS
from mpp.mfe.interfaces.features import RD, DTIFeatures
from mpp.mfe.utilities import SimgCmd
import mpp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal neuroimaging feature extraction",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
    required = parser.add_argument_group("required arguments")
    required.add_argument("dataset", type=str, help="Dataset (HCP-YA, HCP-A, HCP-D, ABCD)")
    required.add_argument("sublist", type=Path, help="Absolute path to subject ID list file")
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "--diff_dir", type=Path, default=None,
        help="Directory containing preprocessed diffusion files")
    optional.add_argument("--work_dir", type=Path, default=Path.cwd(), help="Work directory")
    optional.add_argument("--output_dir", type=Path, default=Path.cwd(), help="Output directory")
    optional.add_argument("--simg", type=Path, default=None, help="singularity image")
    optional.add_argument(
        "--config", type=Path, dest="config", default=files(mpp)/"mfe"/"default.config",
        help="Configuration file for dataset directories")
    optional.add_argument("--condordag", action="store_true", help="Submit as DAG to HTCondor")
    config = vars(parser.parse_args())

    # Set-up
    simg_cmd = SimgCmd(config)
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    config["tmp_dir"] = Path(config["work_dir"], f"mfe_dti_{config['dataset']}_tmp")
    config["tmp_dir"].mkdir(parents=True, exist_ok=True)
    config_parse = configparser.ConfigParser()
    config_parse.read(config["config"])
    config.update({option: config_parse["USER"][option] for option in config_parse["USER"]})
    sublist = pd.read_csv(config["sublist"], header=None, dtype=str).squeeze("columns").tolist()
    mfe_wf = pe.Workflow(f"mfe_dti_{config['dataset']}_wf", base_dir=config["work_dir"])
    mfe_wf.config["execution"]["try_hard_link_datasink"] = "false"
    mfe_wf.config["execution"]["crashfile_format"] = "txt"
    mfe_wf.config["execution"]["stop_on_first_crash"] = "true"

    init_data = pe.Node(
        InitDTIData(config=config), "init_data", iterables=[("subject", sublist)])
    dtifit = pe.Node(fsl.DTIFit(command=simg_cmd.cmd("dtifit")), "dtifit")
    rd = pe.Node(RD(config=config), "rd")
    tbss = pe.JoinNode(
        TBSS(config=config, simg_cmd=simg_cmd), "tbss", joinsource="init_data",
        joinfield=["fa_files", "md_files", "ad_files", "rd_files", "subjects"])
    features = pe.Node(DTIFeatures(config=config, sublist=sublist), "features")
    save_features = pe.Node(SaveDTIFeatures(config=config), "save_features")
    mfe_wf.connect([
        (init_data, dtifit, [
            ("dwi", "dwi"), ("bvals", "bvals"), ("bvecs", "bvecs"), ("mask", "mask")]),
        (init_data, rd, [("subject", "subject")]),
        (dtifit, rd, [("L2", "l2_file"), ("L3", "l3_file")]),
        (init_data, tbss, [("subject", "subjects"), ("dataset_dir", "dataset_dir")]),
        (dtifit, tbss, [("FA", "fa_files"), ("MD", "md_files"), ("L1", "ad_files")]),
        (rd, tbss, [("rd_file", "rd_files")]),
        (tbss, features, [
            ("fa_skeleton_file", "fa_skeleton_file"), ("md_skeleton_file", "md_skeleton_file"),
            ("ad_skeleton_file", "ad_skeleton_file"), ("rd_skeleton_file", "rd_skeleton_file")]),
        (features, save_features, [("dti_features", "features")])])

    # Run workflow
    mfe_wf.write_graph()
    if config["condordag"]:
        mfe_wf.run(
            plugin="CondorDAGMan",
            plugin_args={
                "dagman_args": f"-outfile_dir {config['tmp_dir']} -import_env",
                "wrapper_cmd": files(mpp) / "venv_wrapper.sh",
                "override_specs": "request_memory = 10 GB\nrequest_cpus = 1"})
    else:
        mfe_wf.run()


if __name__ == '__main__':
    main()
