from importlib.resources import files
from pathlib import Path
from shutil import copyfile
import logging

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import datalad.api as dl
import pandas as pd

from mpp.exceptions import DatasetError
import mpp

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
    t1_to_mni = traits.File(exists=True, desc="T1-to-MNI transform")


class InitData(SimpleInterface):
    """Install and get subject-specific data"""
    input_spec = _InitDataInputSpec
    output_spec = _InitDataOutputSpec

    def _run_interface(self, runtime):
        root_data_dir = Path(self.inputs.config["tmp_dir"], self.inputs.config["subject"])
        param = self.inputs.config["param"]

        self._results["dataset_dir"] = root_data_dir
        self._results["rs_runs"] = param["rests"]
        self._results["t_runs"] = param["tasks"]
        self._results["data_files"] = {}

        dl.install(root_data_dir, source=param["url"])
        dl.get(param["dir"], dataset=root_data_dir, get_data=False)
        dl.get(param["sub_dir"], dataset=param["dir"], get_data=False)

        if self.inputs.config["dataset"] in ["HCP-YA", "HCP-A", "HCP-D"]:
            mni_dir = Path(param["sub_dir"], "MNINonLinear")
            func_dir = Path(mni_dir, "Results")
            anat_dir = Path(param["sub_dir"], "T1w")
            fs_dir = Path(anat_dir, self.inputs.config["subject"])
            self._results["anat_dir"] = anat_dir
            self._results["fs_dir"] = fs_dir
            dl.get(mni_dir, dataset=param["sub_dir"], get_data=False)
            dl.get(anat_dir, dataset=param["sub_dir"], get_data=False)

            # required for computing confounds
            if self.inputs.config["dataset"] in ["HCP-A", "HCP-D"]:
                    astats = Path(fs_dir, "stats", "aseg.stats")
                    dl.get(astats, dataset=anat_dir)
                    self._results["data_files"]["astats"] = astats

            if "rfMRI" in self.inputs.config["modality"]:
                rs_files = {"atlas_mask": Path(mni_dir, "ROIs", "Atlas_wmparc.2.nii.gz")}
                for run in param["rests"]:
                    rs_files[f"{run}_surf"] = Path(
                        func_dir, run, f"{run}_Atlas_MSMAll_{param['clean']}.dtseries.nii")
                    rs_files[f"{run}_vol"] = Path(func_dir, run, f"{run}_{param['clean']}.nii.gz")
                self._results["hcpd_b_runs"] = 0
                hcp_b_runs = {}
                for key, val in rs_files.items():
                    if val.is_symlink():
                        dl.get(val, dataset=mni_dir)
                    else:
                        if self.inputs.config["dataset"] == "HCP-D":
                            if "AP" in str(val):
                                rs_file_a = Path(str(val).replace("_AP", "a_AP"))
                                rs_file_b = Path(str(val).replace("_AP", "b_AP"))
                            elif "PA" in str(val):
                                rs_file_a = Path(str(val).replace("_PA", "a_PA"))
                                rs_file_b = Path(str(val).replace("_PA", "b_PA"))
                            else:
                                raise ValueError("file name %s has neither PA nor AP", val)
                            rs_files[key] = ""
                            if rs_file_a.is_symlink:
                                rs_files[key] = rs_file_a
                                dl.get(rs_file_a, dataset=mni_dir)
                            if rs_file_b.is_symlink():
                                hcp_b_runs[f"{key}b"] = rs_file_b
                                dl.get(rs_file_b, dataset=mni_dir)
                                self._results["hcpd_b_runs"] = self._results["hcpd_b_runs"] + 1
                        else:
                            rs_files[key] = ""
                self._results["data_files"].update(rs_files)
                self._results["data_files"].update(hcp_b_runs)

            if "tfMRI" in self.inputs.config["modality"]:
                t_files = {"atlas_mask": Path(mni_dir, "ROIs", "Atlas_wmparc.2.nii.gz")}
                for run in param["tasks"]:
                    t_files[f"{run}_surf"] = Path(func_dir, run, f"{run}_Atlas_MSMAll.dtseries.nii")
                    t_files[f"{run}_vol"] = Path(func_dir, run, f"{run}.nii.gz")
                    ev_files = param["ev_files"][f"tfMRI_{run.split('_')[1]}"]
                    t_files[f"{run}_ev"] = []
                    for ev_file in ev_files:
                        t_files[f"{run}_ev"].append(Path(func_dir, run, "EVs", ev_file))
                for key, val in t_files.items():
                    dl.get(val, dataset=mni_dir)
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
                        dl.get(val, dataset=mni_dir)
                    else:
                        dl.get(val, dataset=anat_dir)
                self._results["data_files"].update(s_files)

            if "dMRI" in self.inputs.config["modality"]:
                if self.inputs.config["dataset"] == "HCP-YA":
                    d_dir = Path(param["sub_dir"], "T1w", "Diffusion")
                    dl.get(d_dir.parent, dataset=param["sub_dir"], get_data=False)
                else:
                    d_data_dir = Path(
                        self.inputs.config["tmp_dir"], f"{self.inputs.config['subject']}_diff")
                    dl.install(d_data_dir, source=param["diff_url"])
                    d_dir = Path(d_data_dir, self.inputs.config["subject"])
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
                    "rh_thickness": Path(fs_dir, "surf", "rh.thickness"),
                    "brain_mask": Path(fs_dir, "mri", "brainmask.mgz")}
                for key, val in fs_files.items():
                    dl.get(val, dataset=anat_dir)
                self._results["data_files"].update(fs_files)

                anat_files = {
                    "t1_restore_brain": Path(anat_dir, "T1w_acpc_dc_restore_brain.nii.gz")}
                for key, val in anat_files.items():
                    dl.get(val, dataset=anat_dir)
                self._results["data_files"].update(anat_files)

                t1mni_file = Path(mni_dir, "xfms", "acpc_dc2standard.nii.gz")
                dl.get(t1mni_file, dataset=mni_dir)
                self._results["data_files"]["t1_to_mni"] = t1mni_file

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
                self._results["t1_to_mni"] = self._results["data_files"]["t1_to_mni"]

        else:
            raise DatasetError()

        return runtime


class _PickAtlasInputSpec(BaseInterfaceInputSpec):
    level = traits.Str(mandatory=True, desc="granularity level")


class _PickAtlasOutputSpec(TraitedSpec):
    level = traits.Str(desc="granularity level")
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
        data_dir = files(mpp) / "data"
        self._results["parc_sch"] = Path(
            data_dir, f"Schaefer2018_{self.inputs.level}00Parcels_17Networks_order.dlabel.nii")
        self._results["parc_mel"] = Path(
            data_dir, f"Tian_Subcortex_S{self.inputs.level}_3T.nii.gz")
        self._results["lh_annot"] = Path(
            data_dir, f"lh.Schaefer2018_{self.inputs.level}00Parcels_17Networks_order.annot")
        self._results["rh_annot"] = Path(
            data_dir, f"rh.Schaefer2018_{self.inputs.level}00Parcels_17Networks_order.annot")

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
        copyfile(self.inputs.lh_annot, Path(out_dir, Path(self.inputs.lh_annot).name))
        copyfile(self.inputs.rh_annot, Path(out_dir, Path(self.inputs.rh_annot).name))
        annot_name = str(Path(self.inputs.lh_annot).name).split(".")[1:-1]
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
            if pheno_data[val].values.shape[0]:
                self._results["pheno"][key] = pheno_data[val].values[0]
            else:
                self._results["pheno"][key] = 999 # missing value

        return runtime


class _SaveFeaturesInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    s_rsfc = traits.Dict(dtype=float, desc="resting-state functional connectivity")
    d_rsfc = traits.Dict(dtype=float, desc="dynamic functional connectivity")
    e_rsfc = traits.Dict(dtype=float, desc="resting-state effecgive connectivity")
    rs_stats = traits.Dict(dtype=float, desc="network statistics")
    s_tfc = traits.Dict({}, dtype=dict, desc="static task-based functional connectivity")
    e_tfc = traits.Dict({}, dtype=dict, desc="task-based effective connectivity")
    myelin = traits.Dict(dtype=float, desc="myelin content estimates")
    morph = traits.Dict(dtype=float, desc="morphometry features")
    sc_count = traits.Dict(desc="structural connectome based on streamline counts")
    sc_length = traits.Dict(desc="structural connectome based on streamline length")
    conf = traits.Dict(desc="confounding variables")
    pheno = traits.Dict(dtype=float, desc="phenotypes")
    dataset_dir = traits.Directory(mandatory=True, desc="absolute path to installed root dataset")


class SaveFeatures(SimpleInterface):
    """Save extracted features"""
    input_spec = _SaveFeaturesInputSpec

    def _write_data_level(self, level: int, data_in: dict, prefix: str, data_type: str) -> None:
        if "surf" in data_type:
            n_parcel = {1: 100, 2: 200, 3: 300, 4: 400}
        else:
            n_parcel = {1: 116, 2: 232, 3: 350, 4: 454}
        data = {}
        for parcel_i in range(n_parcel[level]):
            prefix_i = f"{prefix}_{parcel_i}"
            if data_type == "conn_sym":
                for parcel_j in range(parcel_i + 1, n_parcel[level]):
                    data[f"{prefix_i}_{parcel_j}"] = data_in[f"level{level}"][parcel_i][parcel_j]
            elif data_type == "conn_asym":
                for parcel_j in range(n_parcel[level]):
                    data[f"{prefix_i}_{parcel_j}"] = data_in[f"level{level}"][parcel_i][parcel_j]
            else:
                data[prefix_i] = data_in[f"level{level}"][parcel_i]

        self._write_data(data, f"{prefix}_level{level}")

    def _write_data(self, data: dict, key: str) -> None:
        data_pd = pd.DataFrame(data, index=[self.inputs.config["subject"]])
        data_pd.to_hdf(self._output, key, mode="a", format="fixed")

    def _run_interface(self, runtime):
        self._output = Path(
            self.inputs.config["output_dir"], self.inputs.config["dataset"],
            f"{self.inputs.config['subject']}.h5")
        if self._output.exists():
            dl.get(self._output, dataset=self.inputs.config["output_dir"])
            dl.unlock(self._output, dataset=self.inputs.config["output_dir"])

        self._write_data(self.inputs.conf, "confound")
        self._write_data(self.inputs.pheno, "phenotype")

        for level in [1, 2, 3, 4]:
            if "rfMRI" in self.inputs.config["modality"]:
                self._write_data_level(level, self.inputs.s_rsfc, "rs_sfc", "conn_sym")
                self._write_data_level(level, self.inputs.d_rsfc, "rs_dfc", "conn_asym")
                self._write_data_level(level, self.inputs.e_rsfc, "rs_ec", "conn_asym")
                for stat in ["cpl", "eff", "mod"]:
                    self._write_data(
                        {stat: self.inputs.rs_stats[stat][f"level{level}"]},
                        f"rs_{stat}_level{level}")
                self._write_data_level(level, self.inputs.rs_stats["par"], "rs_par", "array")

            if "tfMRI" in self.inputs.config["modality"]:
                for key, val in self.inputs.s_tfc.items():
                    self._write_data_level(level, val, f"{key}_sfc", "conn_sym")
                    self._write_data_level(level, self.inputs.e_tfc[key], f"{key}_ec", "conn_asym")

            if "sMRI" in self.inputs.config["modality"]:
                self._write_data_level(level, self.inputs.myelin, "s_myelin", "array")
                self._write_data_level(level, self.inputs.morph["gmv"], "s_gmv", "array")
                for stat in ["cs", "ct"]:
                    self._write_data_level(
                        level, self.inputs.morph[stat], f"s_{stat}", "array_surf")

            if "dMRI" in self.inputs.config["modality"]:
                self._write_data_level(level, self.inputs.sc_count, "d_scc", "conn_asym")
                self._write_data_level(level, self.inputs.sc_length, "d_scl", "conn_asym")

        dl.remove(dataset=self.inputs.dataset_dir, reckless="kill")
        if "dMRI" in self.inputs.config["modality"]:
            if self.inputs.config["dataset"] in ["HCP-A", "HCP-D"]:
                d_data_dir = Path(
                    self.inputs.config["tmp_dir"], f"{self.inputs.config['subject']}_diff")
                dl.remove(dataset=d_data_dir, reckless="kill")

        return runtime


class _SaveDTIFeaturesInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    features = traits.Dict(dtype=float, desc="region-wise DTI features")


class SaveDTIFeatures(SimpleInterface):
    """Save extracted features"""
    input_spec = _SaveDTIFeaturesInputSpec

    def _run_interface(self, runtime):
        self._output = Path(
            self.inputs.config["output_dir"], f"{self.inputs.config['dataset']}_dti.h5")
        for feature, feature_data in self.inputs.features.items():
            for subject, subject_data in feature_data.items():
                data_pd = pd.DataFrame(subject_data)
                data_pd.to_hdf(self._output, f"{feature}_{subject}", mode="a", format="fixed")

        return runtime
