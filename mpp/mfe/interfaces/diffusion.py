from os import chdir
from pathlib import Path
from shutil import copyfile, rmtree
import logging
import subprocess

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import datalad.api as dl

logging.getLogger("datalad").setLevel(logging.WARNING)


class _ProbTractInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    data_files = traits.Dict(mandatory=True, dtype=Path, desc="collection of filenames")
    fs_dir = traits.Directory(desc="FreeSurfer subject directory")
    simg_cmd = traits.Any(mandatory=True, desc="command for using singularity image (or not)")


class _ProbTractOutputSpec(TraitedSpec):
    tck_file = traits.File(exists=True, desc="Tractogram file")


class ProbTract(SimpleInterface):
    """Probabilistic tractography"""
    input_spec = _ProbTractInputSpec
    output_spec = _ProbTractOutputSpec

    def _run_interface(self, runtime):
        subject = self.inputs.config['subject']

        # Fiber orientation
        res_wm_file = Path(self.inputs.config["tmp_dir"], f"{subject}_sfwm.txt")
        res_gm_file = Path(self.inputs.config["tmp_dir"], f"{subject}_gm.txt")
        res_csf_file = Path(self.inputs.config["tmp_dir"], f"{subject}_csf.txt")
        fod_wm_file = Path(self.inputs.config["tmp_dir"], f"{subject}_fod_wm.mif")
        args = [
            "-shells", self.inputs.config["param"]["shells"], "-nthreads", "0",
            "-mask", str(self.inputs.data_files["nodif_mask"]),
            "-fslgrad", str(self.inputs.data_files["bvec"]), str(self.inputs.data_files["bval"]),
            str(self.inputs.data_files["dwi"])]
        subprocess.run(
            self.inputs.simg_cmd.cmd("dwi2response").split() + ["dhollander", "-force"] + args
            + [str(res_wm_file), str(res_gm_file), str(res_csf_file)], check=True)
        subprocess.run(
            self.inputs.simg_cmd.cmd("dwi2fod").split() + ["msmt_csd", "-force"] + args
            + [str(res_wm_file), str(fod_wm_file), str(res_gm_file),
               str(Path(self.inputs.config["tmp_dir"], f"{subject}_fod_gm.mif")), str(res_csf_file),
               str(Path(self.inputs.config["tmp_dir"], f"{subject}_fod_csf.mif"))], check=True)

        ftt_file = Path(self.inputs.config["tmp_dir"], f"{subject}_ftt.nii.gz")
        subprocess.run(
            self.inputs.simg_cmd.cmd('5ttgen').split() + [
                "hsvs", "-nocleanup", "-force", str(self.inputs.fs_dir), str(ftt_file)], check=True)

        # Tracktography
        params = {
            # default parameters of tckgen
            "algorithm": "iFOD2", "step": str(0.5 * self.inputs.config["param"]["diff_res"]),
            "angle": "45", "minlength": str(2 * self.inputs.config["param"]["diff_res"]),
            "cutoff": "0.05", "trials": "1000", "samples": "4", "downsample": "3", "power": "0.33",
            # parameters different from default (Jung et al. 2021)
            "maxlength": "250", "max_attempts_per_seed": "50", "select": "10000000"}
        self._results["tck_file"] = Path(
            self.inputs.config["tmp_dir"], f"{subject}_WBT_10M_ctx.tck")
        tck = self.inputs.simg_cmd.cmd("tckgen").split()
        for param, value in params.items():
            tck = tck + [f"-{param}", value]
        tck = tck + [
            "-seed_dynamic", str(fod_wm_file), "-act", str(ftt_file),
            "-output_seeds", str(
                Path(self.inputs.config["tmp_dir"], f"{subject}_WBT_10M_seeds_ctx.txt")),
            "-backtrack", "-crop_at_gmwmi", "-nthreads", "0", "-force",
            "-fslgrad", str(self.inputs.data_files["bvec"]), str(self.inputs.data_files["bval"]),
            str(fod_wm_file), str(self._results["tck_file"])]
        subprocess.run(tck, check=True)

        return runtime


class _TBSSInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    fa_files = traits.List(mandatory=True, desc="list of FA files")
    md_files = traits.List(mandatory=True, desc="list of MD files")
    ad_files = traits.List(mandatory=True, desc="list of AD files")
    rd_files = traits.List(mandatory=True, desc="list of RD files")
    subjects = traits.List(mandatory=True, desc="list of subjects")
    dataset_dir = traits.Directory(mandatory=True, desc="absolute path to installed root dataset")
    simg_cmd = traits.Any(mandatory=True, desc="command for using singularity image (or not)")


class _TBSSOutputSpec(TraitedSpec):
    fa_skeleton_file = traits.File(exists=True, desc="skeletonised FA file")
    md_skeleton_file = traits.File(exists=True, desc="skeletonised MD file")
    ad_skeleton_file = traits.File(exists=True, desc="skeletonised AD file")
    rd_skeleton_file = traits.File(exists=True, desc="skeletonised RD file")


class TBSS(SimpleInterface):
    """Create group skeleton map based on FA maps"""
    input_spec = _TBSSInputSpec
    output_spec = _TBSSOutputSpec

    def _copy_files(self, files_in: list, dir_out: Path) -> list:
        dir_out.mkdir(parents=True, exist_ok=True)
        files_out = []
        for subject, nonfa_file in zip(self.inputs.subjects, files_in):
            copyfile(nonfa_file, Path(dir_out, f"{subject}_FA.nii.gz"))
            files_out.append(f"{subject}_FA.nii.gz")
        return files_out

    def _run_interface(self, runtime):
        dl.remove(dataset=self.inputs.dataset_dir, reckless="kill")
        fa_dir = Path(self.inputs.config["tmp_dir"], "tbss_fa")
        fa_dir.mkdir(parents=True, exist_ok=True)
        chdir(fa_dir)

        fa_list = self._copy_files(self.inputs.fa_files, fa_dir)
        subprocess.run(self.inputs.simg_cmd.cmd("tbss_1_preproc").split() + fa_list, check=True)
        subprocess.run(self.inputs.simg_cmd.cmd("tbss_2_reg").split() + ["-n"], check=True)
        subprocess.run(self.inputs.simg_cmd.cmd("tbss_3_postreg").split() + ["-S"], check=True)
        subprocess.run(self.inputs.simg_cmd.cmd("tbss_4_prestats").split() + ["0.2"], check=True)
        self._results["fa_skeleton_file"] = Path(fa_dir, "stats", "all_FA_skeletonised.nii.gz")

        files_in_dict = {
            "MD": self.inputs.md_files, "AD": self.inputs.ad_files, "RD": self.inputs.rd_files}
        for file_type, files_in in files_in_dict.items():
            _ = self._copy_files(files_in, Path(fa_dir, file_type))
            subprocess.run(self.inputs.simg_cmd.cmd("tbss_non_FA").split() + [file_type], check=True)
            rmtree(Path(fa_dir, file_type))
        self._results["md_skeleton_file"] = Path(fa_dir, "stats", "all_MD_skeletonised.nii.gz")
        self._results["ad_skeleton_file"] = Path(fa_dir, "stats", "all_AD_skeletonised.nii.gz")
        self._results["rd_skeleton_file"] = Path(fa_dir, "stats", "all_RD_skeletonised.nii.gz")

        return runtime
