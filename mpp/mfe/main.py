from pathlib import Path
import argparse

import nipype.pipeline as pe
from nipype.interfaces import fsl, freesurfer
from nipype.interfaces import utility as niu

from mpp.mfe.interfaces.data import InitData, PickAtlas, SubDirAnnot, SaveFeatures, Phenotypes
from mpp.mfe.interfaces.diffusion import ProbTract
from mpp.mfe.interfaces.features import FC, NetworkStats, Anat, SC, Confounds
from mpp.mfe.utilities import SimgCmd, dataset_params, add_subdir, CombineStrings, CombineAtlas

base_dir = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal neuroimaging feature extraction",
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
    required = parser.add_argument_group("required arguments")
    required.add_argument("dataset", type=str, help="Dataset (HCP-YA, HCP-A, HCP-D, ABCD)")
    required.add_argument("subject", type=str, help="Subject ID")
    required.add_argument(
        "--modality", nargs="+", required=True,
        help="List of modalities (rfMRI, tfMRI, sMRI, dMRI)")
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "--diff_dir", type=Path, default=None,
        help="Directory containing preprocessed diffusion files")
    optional.add_argument(
        "--pheno_dir", type=Path, default=None, help="Directory containing phenotype data")
    optional.add_argument("--work_dir", type=Path, default=Path.cwd(), help="Work directory")
    optional.add_argument("--output_dir", type=Path, default=Path.cwd(), help="Output directory")
    optional.add_argument("--simg", type=Path, default=None, help="singularity image")
    optional.add_argument("--condordag", action="store_true", help="Submit as DAG to HTCondor")
    config = vars(parser.parse_args())

    # Set-up
    simg_cmd = SimgCmd(config)
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    config["param"] = dataset_params(
        config["dataset"], Path(config["work_dir"], config["subject"]), config["pheno_dir"],
        config["subject"])
    mfe_wf = pe.Workflow(f"mfe_{config['subject']}_wf", base_dir=config["work_dir"])
    mfe_wf.config["execution"]["try_hard_link_datasink"] = "false"
    mfe_wf.config["execution"]["crashfile_format"] = "txt"
    mfe_wf.config["execution"]["stop_on_first_crash"] = "true"

    # Input and output
    init_data = pe.Node(InitData(config=config, simg_cmd=simg_cmd), "init_data")
    save_features = pe.Node(SaveFeatures(config=config), "save_features")
    mfe_wf.connect([(init_data, save_features, [("dataset_dir", "dataset_dir")])])

    # rfMRI features
    if "rfMRI" in config["modality"]:
        rsfc = pe.Node(FC(config=config, modality="rfMRI"), "rsfc")
        rs_stats = pe.Node(NetworkStats(), "rs_stats")
        mfe_wf.connect([
            (init_data, rsfc, [
                ("rs_runs", "rs_runs"), ("data_files", "data_files"), ("hcpd_b_runs", "hcpd_b_runs")]),
            (rsfc, rs_stats, [("sfc", "conn")]),
            (rsfc, save_features, [("sfc", "s_rsfc"), ("dfc", "d_rsfc")]),
            (rs_stats, save_features, [("stats", "rs_stats")])])

    # tfMRI features
    if "tfMRI" in config["modality"]:
        tfc = pe.Node(FC(config=config, modality="tfMRI"), "tfc")
        mfe_wf.connect([
            (init_data, tfc, [("t_runs", "t_runs"), ("data_files", "data_files")]),
            (tfc, save_features, [("sfc", "tfc")])])

    # sMRI features
    if "sMRI" in config["modality"]:
        anat = pe.Node(Anat(config=config, simg_cmd=simg_cmd), "anat")
        mfe_wf.connect([
            (init_data, anat, [("data_files", "data_files"), ("anat_dir", "anat_dir")]),
            (anat, save_features, [("myelin", "myelin"), ("morph", "morph")])])

    # dMRI features
    if "dMRI" in config["modality"]:
        tmp_dir = Path(config["work_dir"], "sc_tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        fs_opt = f"--env SUBJECTS_DIR={tmp_dir}"
        sub_dir = pe.Node(niu.Function(function=add_subdir, output_names=["sub_dir"]), "sub_dir")
        sub_dir.inputs.sub_dir = tmp_dir
        sub_dir.inputs.subject = config["subject"]
        pick_atlas = pe.Node(PickAtlas(), "pick_atlas", iterables=[("level", [0, 1, 2, 3])])
        add_annot = pe.Node(SubDirAnnot(config=config), "add_annot")
        aseg = pe.Node(
            CombineStrings(input1=str(tmp_dir), input2="/aparc2aseg_", input4=".nii.gz"), "aseg")
        mfe_wf.connect([
            (init_data, sub_dir, [("fs_dir", "fs_dir")]),
            (sub_dir, add_annot, [("sub_dir", "sub_dir")]),
            (pick_atlas, aseg, [("level", "input3")])])

        # Melbourne subcortex atlas to subject T1 space
        std2t1 = pe.Node(fsl.InvWarp(command=simg_cmd.cmd("invwarp")), "std2t1")
        mel_t1 = pe.Node(
            fsl.ApplyWarp(command=simg_cmd.cmd("applywarp"), interp="nn", relwarp=True), "mel_t1")
        mfe_wf.connect([
            (init_data, std2t1, [("talairach_xfm", "warp"), ("t1_restore_brain", "reference")]),
            (init_data, mel_t1, [("t1_restore_brain", "ref_file")]),
            (pick_atlas, mel_t1, [("parc_mel", "in_file")]),
            (std2t1, mel_t1, [("inverse_warp", "field_file")])])

        # Schaefer cortex atlas to aseg in subject T1 space
        lannot_sub = pe.Node(freesurfer.SurfaceTransform(
            command=simg_cmd.cmd("mri_surf2surf", options=fs_opt), hemi="lh",
            source_subject="fsaverage", target_subject=config["subject"]), "lannot_sub")
        rannot_sub = pe.Node(freesurfer.SurfaceTransform(
            command=simg_cmd.cmd("mri_surf2surf", options=fs_opt), hemi="rh",
            source_subject="fsaverage", target_subject=config["subject"]), "rannot_sub")
        sch_aseg = pe.Node(freesurfer.Aparc2Aseg(
            command=simg_cmd.cmd("mri_aparc2aseg", options=fs_opt), subject_id=config["subject"]),
            "sch_aseg")
        sch_t1 = pe.Node(
            fsl.FLIRT(command=simg_cmd.cmd("flirt"), interp="nearestneighbour"), "sch_t1")
        mfe_wf.connect([
            (sub_dir, lannot_sub, [("sub_dir", "subjects_dir")]),
            (pick_atlas, lannot_sub, [("lh_annot", "source_annot_file")]),
            (sub_dir, rannot_sub, [("sub_dir", "subjects_dir")]),
            (pick_atlas, rannot_sub, [("rh_annot", "source_annot_file")]),
            (lannot_sub, add_annot, [("out_file", "lh_annot")]),
            (rannot_sub, add_annot, [("out_file", "rh_annot")]),
            (sub_dir, sch_aseg, [("sub_dir", "subjects_dir")]),
            (init_data, sch_aseg, [
                ("lh_aparc", "lh_annotation"), ("rh_aparc", "rh_annotation"),
                ("lh_pial", "lh_pial"), ("rh_pial", "rh_pial"), ("lh_ribbon", "lh_ribbon"),
                ("rh_ribbon", "rh_ribbon"), ("lh_white", "lh_white"), ("rh_white", "rh_white"),
                ("ribbon", "ribbon")]),
            (add_annot, sch_aseg, [("args", "args")]),
            (aseg, sch_aseg, [("output", "out_file")]),
            (init_data, sch_t1, [("t1_restore_brain", "reference")]),
            (sch_aseg, sch_t1, [("out_file", "in_file")])])

        # Combine atlases & downsample
        combine = pe.Node(CombineAtlas(config=config), "combine")
        downsamp = pe.Node(fsl.FLIRT(
            command=simg_cmd.cmd("flirt"), interp="nearestneighbour", datatype="int",
            apply_isoxfm=config["param"]["diff_res"]), "downsamp")
        mfe_wf.connect([
            (pick_atlas, combine, [("level", "level")]),
            (mel_t1, combine, [("out_file", "subcort_file")]),
            (sch_t1, combine, [("out_file", "cort_file")]),
            (combine, downsamp, [("combined_file", "in_file"), ("combined_file", "reference")])])

        # Tractography & SC
        prob_track = pe.Node(ProbTract(config=config, simg_cmd=simg_cmd), "prob_track")
        sc = pe.JoinNode(
            SC(config=config, simg_cmd=simg_cmd), "sc", joinfield=["atlas_files"],
            joinsource="pick_atlas")
        mfe_wf.connect([
            (init_data, prob_track, [("data_files", "data_files"), ("fs_dir", "fs_dir")]),
            (init_data, sc, [("data_files", "data_files"), ("fs_dir", "fs_dir")]),
            (downsamp, sc, [("out_file", "atlas_files")]),
            (prob_track, sc, [("tck_file", "tck_file")]),
            (sc, save_features, [("sc_count", "sc_count"), ("sc_length", "sc_length")])])

    # Compute effective connectivity if both functional and diffusion features are requested
    if "rfMRI" in config["modality"] and "dMRI" in config["modality"]:
        mfe_wf.connect([(sc, rsfc, [("sc_count", "sc_count")])])
    if "tfMRI" in config["modality"] and "dMRI" in config["modality"]:
        mfe_wf.connect([sc, tfc, [("sc_count", "sc_count")]])

    # Confounds and phenotypes are always computed
    conf = pe.Node(Confounds(config=config, simg_cmd=simg_cmd), "conf")
    pheno = pe.Node(Phenotypes(config=config), "pheno")
    mfe_wf.connect([
        (init_data, conf, [("data_files", "data_files")]),
        (conf, save_features, [("conf", "conf")]),
        (pheno, save_features, [("pheno", "pheno")])])

    # Run workflow
    mfe_wf.write_graph()
    if config["condordag"]:
        mfe_wf.run(
            plugin="CondorDAGMan",
            plugin_args={
                "dagman_args": f"-outfile_dir {config['work_dir']} -import_env",
                "wrapper_cmd": Path(base_dir, "utilities", "venv_wrapper.sh"),
                "override_specs": "request_memory = 10 GB\nrequest_cpus = 1"})
    else:
        mfe_wf.run()


if __name__ == '__main__':
    main()
