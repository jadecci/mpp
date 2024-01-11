from pathlib import Path
from shutil import copyfile
import logging

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import datalad.api as dl
import pandas as pd
import numpy as np

from mpp.exceptions import DatasetError
from mpp.utilities.data import write_h5

base_dir = Path(__file__).resolve().parent
logging.getLogger("datalad").setLevel(logging.WARNING)


class _InitDataInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    simg_cmd = traits.Any(mandatory=True, desc="command for using singularity image (or not)")


class _InitDataOutputSpec(TraitedSpec):
    fs_dir = traits.Directory(desc="absolute path to the FreeSurfer output directory")
    anat_dir = traits.Directory(desc="absolute path to installed subject T1w directory")
    rs_runs = traits.List(desc="resting-state run names")
    t_runs = traits.List(desc="task run names")
    data_files = traits.Dict(dtype=Path, desc="collection of files")
    hcpd_b_runs = traits.Int(desc="number of HCP-D b runs")
    dataset_dir = traits.Directory(desc="absolute path to installed root dataset")
    talairach_xfm = traits.File(exists=True, desc="Talairach transform")
    t1_restore_brain = traits.File(exists=True, desc="T1 restored brain file")
    lh_aparc = traits.File(exists=True, desc="left aparc annot")
    rh_aparc = traits.File(exists=True, desc="right aparc annot")
    lh_pial = traits.File(exists=True, desc="left pial surface")
    rh_pial = traits.File(exists=True, desc="right pial surface")
    lh_ribbon = traits.File(exists=True, desc="left ribbon")
    rh_ribbon = traits.File(exists=True, desc="right ribbon")
    lh_white = traits.File(exists=True, desc="left white surface")
    rh_white = traits.File(exists=True, desc="right white surface")
    ribbon = traits.File(exists=True, desc="ribbon in volume")


class InitData(SimpleInterface):
    """Install and get subject-specific data"""
    input_spec = _InitDataInputSpec
    output_spec = _InitDataOutputSpec

    def _run_interface(self, runtime):
        root_data_dir = Path(self.inputs.config["work_dir"], self.inputs.config["subject"])
        param = self.inputs.config["param"]

        self._results["dataset_dir"] = root_data_dir
        self._results["rs_runs"] = param["rests"]
        self._results["t_runs"] = param["tasks"]
        self._results["data_files"] = {}

        dl.install(root_data_dir, source=param["url"])
        dl.get(param["dir"], dataset=root_data_dir, get_data=False, source=param["source"])
        dl.get(param["sub_dir"], dataset=param["dir"], get_data=False, source=param["source"])

        if self.inputs.config["dataset"] in ["HCP-YA", "HCP-A", "HCP-D"]:
            mni_dir = Path(param["sub_dir"], "MNINonLinear")
            func_dir = Path(mni_dir, "Results")
            anat_dir = Path(param["sub_dir"], "T1w")
            fs_dir = Path(anat_dir, self.inputs.config["subject"])
            d_dir = self.inputs.config["diff_dir"]
            self._results["anat_dir"] = anat_dir
            self._results["fs_dir"] = fs_dir
            dl.get(mni_dir, dataset=param["sub_dir"], get_data=False, source=param["source"])
            dl.get(anat_dir, dataset=param["sub_dir"], get_data=False, source=param["source"])

            if "rfMRI" in self.inputs.config["modality"]:
                rs_files = {"atlas_mask": Path(mni_dir, "ROIs", "Atlas_wmparc.2.nii.gz")}
                for run in param["rests"]:
                    rs_files[f"{run}_surf"] = Path(
                        func_dir, run, f"{run}_Atlas_MSMAll_{param['clean']}.dtseries.nii")
                    rs_files[f"{run}_vol"] = Path(func_dir, run, f"{run}_{param['clean']}.nii.gz")
                self._results["hcpd_b_runs"] = 0
                for key, val in rs_files.items():
                    if val.is_symlink():
                        dl.get(val, dataset=mni_dir, source=param["source"])
                    else:
                        if self.inputs.dataset == "HCP-D":
                            if "AP" in val:
                                rs_file_a = Path(str(val).replace("_AP", "a_AP"))
                                rs_file_b = Path(str(val).replace("_AP", "b_AP"))
                            elif "PA" in val:
                                rs_file_a = str(val).replace("_PA", "a_PA")
                                rs_file_b = str(val).replace("_PA", "b_PA")
                            else:
                                raise ValueError("file name %s has neither PA nor AP", val)
                            rs_files[key] = ""
                            if rs_file_a.is_symlink:
                                rs_files[key] = rs_file_a
                                dl.get(rs_file_a, dataset=mni_dir, source=param["source"])
                            if rs_file_b.is_symlink():
                                rs_files[f"{key}b"] = rs_file_b
                                dl.get(rs_file_b, dataset=mni_dir, source=param["source"])
                                self._results["hcpd_b_runs"] = self._results["hcpd_b_runs"] + 1
                        else:
                            rs_files[key] = ""
                self._results["data_files"].update(rs_files)

            if "tfMRI" in self.inputs.config["modality"]:
                t_files = {"atlas_mask": Path(mni_dir, "ROIs", "Atlas_wmparc.2.nii.gz")}
                for run in param["tasks"]:
                    t_files[f"{run}_surf"] = Path(func_dir, run, f"{run}_Atlas_MSMAll.dtseries.nii")
                    t_files[f"{run}_vol"] = Path(func_dir, run, f"{run}.nii.gz")
                for key, val in t_files.items():
                    dl.get(val, dataset=mni_dir, source=param["source"])
                self._results["data_files"].update(t_files)

            if "sMRI" in self.inputs.config["modality"]:
                s_files = {
                    "t1_vol": Path(mni_dir, "T1w.nii.gz"),
                    "myelin_l": Path(
                        mni_dir, "fsaverage_LR32k",
                        f"{self.inputs.config['subject']}.L.MyelinMap.32k_fs_LR.func.gii"),
                    "myelin_r": Path(
                        mni_dir, 'fsaverage_LR32k',
                        f"{self.inputs.config['subject']}.R.MyelinMap.32k_fs_LR.func.gii"),
                    "wm_vol": Path(fs_dir, "mri", "wm.mgz"),
                    "white_l": Path(fs_dir, "surf", "lh.white"),
                    "white_r": Path(fs_dir, "surf", "rh.white"),
                    "pial_l": Path(fs_dir, "surf", "lh.pial"),
                    "pial_r": Path(fs_dir, "surf", "rh.pial"),
                    "lh_reg": Path(fs_dir, "surf", "lh.sphere.reg"),
                    "rh_reg": Path(fs_dir, "surf", "rh.sphere.reg"),
                    "ct_l": Path(fs_dir, "surf", "lh.thickness"),
                    "ct_r": Path(fs_dir, "surf", "rh.thickness"),
                    "label_l": Path(fs_dir, "label", "lh.cortex.label"),
                    "label_r": Path(fs_dir, "label", "rh.cortex.label"),
                    "myelin_vol": Path(anat_dir, "T1wDividedByT2w.nii.gz")}
                for key, val in s_files.items():
                    if key == "t1_vol" or key == "myelin_l" or key == "myelin_r":
                        dl.get(val, dataset=mni_dir, source=param["source"])
                    else:
                        dl.get(val, dataset=anat_dir, source=param["source"])
                self._results["data_files"].update(s_files)

            if "dMRI" in self.inputs.config["modality"]:
                if self.inputs.config["dataset"] == "HCP-YA":
                    dl.get(d_dir.parent, param["sub_dir"], get_data=False, source=param["source"])
                d_files = {
                    "dwi": Path(d_dir, "data.nii.gz"), "bval": Path(d_dir, "bvals"),
                    "bvec": Path(d_dir, "bvecs"),
                    "nodif_mask": Path(d_dir, "nodif_brain_mask.nii.gz")}
                for key, val in d_files.items():
                    dl.get(val, dataset=d_dir.parent)
                self._results["data_files"].update(d_files)

                fs_files = {
                    "lh_aparc": Path(fs_dir, "label", "lh.aparc.annot"),
                    "rh_aparc": Path(fs_dir, "label", "rh.aparc.annot"),
                    "lh_pial": Path(fs_dir, "surf", "lh.pial"),
                    "rh_pial": Path(fs_dir, "surf", "rh.pial"),
                    "lh_white": Path(fs_dir, "surf", "lh.white"),
                    "rh_white": Path(fs_dir, "surf", "rh.white"),
                    "lh_reg": Path(fs_dir, "surf", "lh.sphere.reg"),
                    "rh_reg": Path(fs_dir, "surf", "rh.sphere.reg"),
                    "lh_cort_label": Path(fs_dir, "label", "lh.cortex.label"),
                    "rh_cort_label": Path(fs_dir, "label", "rh.cortex.label"),
                    "lh_ribbon": Path(fs_dir, "mri", "lh.ribbon.mgz"),
                    "rh_ribbon": Path(fs_dir, "mri", "rh.ribbon.mgz"),
                    "ribbon": Path(fs_dir, "mri", "ribbon.mgz"),
                    "aseg": Path(fs_dir, "mri", "aseg.mgz"),
                    "aparc_aseg": Path(fs_dir, "mri", "aparc+aseg.mgz"),
                    "talaraich_xfm": Path(fs_dir, "mri", "transforms", "talairach.xfm"),
                    "norm": Path(fs_dir, "mri", "norm.mgz"),
                    "lh_thickness": Path(fs_dir, "surf", "lh.thickness"),
                    "rh_thickness": Path(fs_dir, "surf", "rh.thickness")}
                for key, val in fs_files.items():
                    dl.get(val, dataset=anat_dir, source=param["source"])
                self._results["data_files"].update(fs_files)

                anat_files = {
                    "t1_restore_brain": Path(anat_dir, "T1w_acpc_dc_restore_brain.nii.gz")}
                for key, val in anat_files.items():
                    dl.get(val, dataset=anat_dir, source=param["source"])
                self._results["data_files"].update(anat_files)

                mni_files = {
                    "t1_to_mni": Path(mni_dir, "xfms", "acpc_dc2standard.nii.gz"),
                    "aseg": Path(fs_dir, "mri", "aseg.mgz"),
                    "aparc_aseg": Path(fs_dir, "mri", "aparc+aseg.mgz"),
                    "talairach_xfm": Path(fs_dir, "mri", "xfms", "talairach.xfm"),
                    "norm": Path(fs_dir, "mri", "norm.mgz")}
                for key, val in mni_files.items():
                    dl.get(val, dataset=mni_dir, source=param["source"])
                self._results["data_files"].update(mni_files)

                self._results["talairach_xfm"] = self._results["data_files"]["talairach_xfm"]
                self._results["t1_restore_brain"] = self._results["data_files"]["t1_restore_brain"]
                self._results["lh_aparc"] = self._results["data_files"]["lh_aparc"]
                self._results["rh_aparc"] = self._results["data_files"]["rh_aparc"]
                self._results["lh_pial"] = self._results["data_files"]["lh_pial"]
                self._results["rh_pial"] = self._results["data_files"]["rh_pial"]
                self._results["lh_ribbon"] = self._results["data_files"]["lh_ribbon"]
                self._results["rh_ribbon"] = self._results["data_files"]["rh_ribbon"]
                self._results["lh_white"] = self._results["data_files"]["lh_white"]
                self._results["rh_white"] = self._results["data_files"]["rh_white"]
                self._results["ribbon"] = self._results["data_files"]["ribbon"]

            if "conf" in self.inputs.config["modality"]:
                if self.inputs.config["dataset"] in ["HCP-A", "HCP-D"]:
                    astats = Path(fs_dir, "stats", "aseg.stats")
                    dl.get(astats, dataset=anat_dir, source=param["source"])
                    self._results["data_files"]["astats"] = astats

        else:
            raise DatasetError()

        return runtime


class _PickAtlasInputSpec(BaseInterfaceInputSpec):
    level = traits.Int(mandatory=True, desc="granularity level")


class _PickAtlasOutputSpec(TraitedSpec):
    level = traits.Int(desc="granularity level")
    parc_sch = traits.File(exists=True, desc="Schaefer cortex atlas")
    parc_mel = traits.File(exists=True, desc="Melbourne subcortex atlas")
    lh_annot = traits.File(exists=True, desc="left annot of Schafer cortex atlas")
    rh_annot = traits.File(exists=True, desc="right annot of Schafer cortex atlas")


class PickAtlas(SimpleInterface):
    """Pick out atlases by granularity level"""
    input_spec = _PickAtlasInputSpec
    output_spec = _PickAtlasOutputSpec

    def _run_interface(self, runtime):
        self._results["level"] = self.inputs.level
        data_dir = Path(base_dir, "data")
        self._results["parc_sch"] = Path(
            data_dir, f"Schaefer2018_{self.inputs.level+1}00Parcels_17Networks_order.dlabel.nii")
        self._results["parc_mel"] = Path(
            data_dir, f"Tian_Subcortex_S{self.inputs.level+1}_3T.nii.gz")
        self._results["lh_annot"] = Path(
            data_dir, f"lh.Schaefer2018_{self.inputs.level+1}00Parcels_17Networks_order.annot")
        self._results["rh_annot"] = Path(
            data_dir, f"rh.Schaefer2018_{self.inputs.level+1}00Parcels_17Networks_order.annot")

        return runtime


class _SubDirAnnotInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    sub_dir = traits.Directory(mandatory=True, desc="subjects directory")
    lh_annot = traits.File(mandatory=True, exists=True, desc="left annot of Schafer cortex atlas")
    rh_annot = traits.File(mandatory=True, exists=True, desc="right annot of Schafer cortex atlas")


class _SubDirAnnotOutputSpec(TraitedSpec):
    args = traits.Str(desc="argument for FreeSurfer")


class SubDirAnnot(SimpleInterface):
    """Create annot files in subjects directory and forms an argument for FreeSurfer"""
    input_spec = _SubDirAnnotInputSpec
    output_spec = _SubDirAnnotOutputSpec

    def _run_interface(self, runtime):
        out_dir = Path(self.inputs.sub_dir, self.inputs.config["subject"], "label")
        copyfile(self.inputs.lh_annot, Path(out_dir, self.inputs.lh_annot.name))
        copyfile(self.inputs.rh_annot, Path(out_dir, self.inputs.rh_annot.name))
        annot_name = str(self.inputs.lh_annot.name).split(".")[1:-1]
        annot_name = ".".join(annot_name)
        self._results["args"] = f"--annot {annot_name}"

        return runtime


class _PhenotypesInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")


class _PhenotypesOutputSpec(TraitedSpec):
    pheno = traits.Dict(dtype=float, desc="phenotypes")


class Phenotypes(SimpleInterface):
    """Extract phenotype data"""
    input_spec = _PhenotypesInputSpec
    output_spec = _PhenotypesOutputSpec

    def _run_interface(self, runtime):
        self._results["pheno"] = dict.fromkeys(self.inputs.config["param"]["col_names"].keys())
        for key, val in self.inputs.config["param"]["col_names"].items():
            if self.inputs.config["dataset"] == "HCP-YA":
                pheno_file = self.inputs.config["param"]["pheno_file"]
                pheno_data = pd.read_csv(
                    pheno_file, usecols=["Subject", val], dtype={"Subject": str, val: float})
                pheno_data = pheno_data.loc[pheno_data["Subject"] == self.inputs.config["subject"]]
            elif self.inputs.config["dataset"] in ["HCP-A", "HCP-D"]:
                pheno_file = self.inputs.config["param"]["pheno_files"][key]
                subject = self.inputs.config["subject"].split("_V1_MR")[0]
                pheno_data = pd.read_table(
                    pheno_file, sep="\t", header=0, skiprows=[1],
                    usecols=[4, self.inputs.config["param"]["pheno_cols"][key]],
                    dtype={"src_subject_id": str, val: float})[["src_subject_id", val]]
                pheno_data = pheno_data.loc[pheno_data["src_subject_id"] == subject]
            else:
                raise DatasetError()
            self._results["pheno"][key] = pheno_data[val].values[0]

        return runtime


class _SaveFeaturesInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    s_rsfc = traits.Dict(dtype=float, desc="resting-state functional connectivity")
    d_rsfc = traits.Dict(dtype=float, desc="dynamic functional connectivity")
    rs_stats = traits.Dict(dtype=float, desc="network statistics")
    tfc = traits.Dict({}, dtype=dict, desc="task-based functional connectivity")
    myelin = traits.Dict(dtype=float, desc="myelin content estimates")
    morph = traits.Dict(dtype=float, desc="morphometry features")
    scc_files = traits.Dict(dtype=Path, desc="SC based on streamline count")
    scl_files = traits.Dict(dtype=Path, desc="SC based on streamline length")
    conf = traits.Dict(desc="confounding variables")
    pheno = traits.Dict(dtype=float, desc="phenotypes")
    dataset_dir = traits.Directory(mandatory=True, desc="absolute path to installed root dataset")


class SaveFeatures(SimpleInterface):
    """Save extracted features"""
    input_spec = _SaveFeaturesInputSpec

    def _run_interface(self, runtime):
        output = Path(self.inputs.config["output_dir"], f"{self.inputs.config['subject']}.h5")

        for level in range(4):
            l_key = f"level{level+1}"

            if "rfMRI" in self.inputs.config["modality"]:
                write_h5(output, f"/s_rsfc/{l_key}", self.inputs.s_rsfc[l_key], True)
                write_h5(output, f"/d_rsfc/{l_key}", self.inputs.d_rsfc[l_key], True)
                for stat in ["strength", "betweenness", "participation", "efficiency"]:
                    write_h5(
                        output, f"/rs_stats/{stat}/{l_key}",
                        self.inputs.rs_stats[f"{l_key}_{stat}"], True)

            if "tfMRI" in self.inputs.config["modality"]:
                for key, _ in self.inputs.tfc.items():
                    write_h5(
                        output, f"/tfc/{key}/{l_key}", self.inputs.tfc[key][l_key], True)

            if "sMRI" in self.inputs.config["modality"]:
                write_h5(output, f"/myelin/{l_key}", self.inputs.myelin[l_key], True)
                for stat in ["GMV", "CS", "CT"]:
                    write_h5(
                        output, f"/morphometry/{stat}/{l_key}",
                        self.inputs.morph[f"{l_key}_{stat}"], True)

            if "dMRI" in self.inputs.config["modality"]:
                write_h5(
                    output, f"/sc_count/{l_key}",
                    pd.read_csv(self.inputs.scc_files[l_key], header=None), True)
                write_h5(
                    output, f"/sc_length/{l_key}",
                    pd.read_csv(self.inputs.scl_files[l_key], header=None), True)

        if "conf" in self.inputs.config["modality"]:
            for key, val in self.inputs.conf.items():
                write_h5(output, f"/confound/{key}", np.array(val), True)

        if "pheno" in self.inputs.config["modality"]:
            for key, val in self.inputs.pheno.items():
                write_h5(output, f"/phenotype/{key}", np.array(val), True)

        dl.remove(dataset=self.inputs.dataset_dir, reckless="kill")

        return runtime
