from pathlib import Path
import subprocess

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits


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
        tmp_dir = Path(self.inputs.config["work_dir"], "probtrack_tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        subject = self.inputs.config['subject']

        # Fiber orientation
        res_wm_file = Path(tmp_dir, f"{subject}_sfwm.txt")
        res_gm_file = Path(tmp_dir, f"{subject}_gm.txt")
        res_csf_file = Path(tmp_dir, f"{subject}_csf.txt")
        fod_wm_file = Path(tmp_dir, f"{subject}_fod_wm.mif")
        args = [
            "-shells", self.inputs.config["param"]["shells"], "-nthreads", "0",
            "-mask", str(self.inputs.data_files["nodif_mask"]),
            "-fslgrad", str(self.inputs.data_files["bvec"]), str(self.inputs.data_files["bval"]),
            str(self.inputs.data_files["dwi"])]
        subprocess.run(
            self.inputs.simg_cmd.cmd("dwi2response").split() + ["dhollander", "-force"] + args
            + [str(res_wm_file), str(res_gm_file), str(res_csf_file)], check=True)
        subprocess.run(
            self.inputs.simg_cmd.cmd("dwi2fod").split() + ["msmt_csd"] + args
            + [str(res_wm_file), str(fod_wm_file), str(res_gm_file),
               str(Path(tmp_dir, f"{subject}_fod_gm.mif")), str(res_csf_file),
               str(Path(tmp_dir, f"{subject}_fod_csf.mif"))], check=True)

        ftt_file = Path(tmp_dir, f"{subject}_ftt.nii.gz")
        subprocess.run(
            self.inputs.simg_cmd.cmd('5ttgen').split() + [
                "hsvs", "-nocleanup", str(self.inputs.fs_dir), str(ftt_file)], check=True)

        # Tracktography
        params = {
            # default parameters of tckgen
            "algorithm": "iFOD2", "step": str(0.5 * self.inputs.config["param"]["voxel_size"]),
            "angle": "45", "minlength": str(2 * self.inputs.config["param"]["voxel_size"]),
            "cutoff": "0.05", "trials": "1000", "samples": "4", "downsample": "3", "power": "0.33",
            # parameters different from default (Jung et al. 2021)
            "maxlength": "250", "max_attempts_per_seed": "50", "select": "10000000"}
        self._results["tck_file"] = Path(tmp_dir, f"{subject}_WBT_10M_ctx.tck")
        tck = self.inputs.simg_cmd.cmd("tckgen").split()
        for param, value in params.items():
            tck = tck + [f"-{param}", value]
        tck = tck + [
            "-seed_dynamic", str(fod_wm_file), "-act", str(ftt_file),
            "-output_seeds", str(Path(tmp_dir, f"{subject}_WBT_10M_seeds_ctx.txt")),
            "-backtrack", "-crop_at_gmwmi", "-nthreads", "0",
            "-fslgrad", str(self.inputs.bvec), str(self.inputs.bval),
            str(self.inputs.fod_wm_file), str(self._results["tck_file"])]
        subprocess.run(tck, check=True)

        return runtime
