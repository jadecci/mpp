from importlib.resources import files
from os import environ
from pathlib import Path
from typing import Optional
import sys
import subprocess

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
from nipype.interfaces import fsl, freesurfer
from rdcmpy import RegressionDCM
from scipy.ndimage import binary_erosion
from scipy.stats import zscore
import bct
import datalad.api as dl
import nibabel as nib
import numpy as np
import pandas as pd

from mpp.exceptions import DatasetError
from mpp.mfe.utilities import AddSubDir
import mpp


class _FCInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    data_files = traits.Dict(mandatory=True, dtype=Path, desc="collection of filenames")
    modality = traits.Str(mandatory=True, desc="modality")
    rs_runs = traits.List(desc="resting-state run names")
    t_runs = traits.List(desc="task run names")
    hcpd_b_runs = traits.Int(0, usedefault=True, desc="number of b runs added for HCP-D subject")
    sc_count = traits.Dict({}, usedefault=True, desc="structural connectome")


class _FCOutputSpec(TraitedSpec):
    sfc = traits.Dict(dtype=float, desc="static FC")
    dfc = traits.Dict(dtype=float, desc="dynamic FC")
    ec = traits.Dict(dtype=float, desc="effective connectivity")


class FC(SimpleInterface):
    """Compute resting-state static and dynamic functional connectivity.
    If structural connectivity was computed, then also estimate effective connectivity"""
    input_spec = _FCInputSpec
    output_spec = _FCOutputSpec

    def _nuisance_conf_hcp(self, t_vol: np.ndarray) -> np.ndarray:
        # Atlas labels follow FreeSurferColorLUT
        # see https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
        atlas_file = self.inputs.data_files["atlas_mask"]
        csf_code = np.array([4, 5, 14, 15, 24, 31, 43, 44, 63, 250, 251, 252, 253, 254, 255]) - 1
        data = t_vol.reshape((t_vol.shape[0] * t_vol.shape[1] * t_vol.shape[2], t_vol.shape[3]))
        atlas = nib.load(atlas_file).get_fdata()
        atlas = atlas.reshape((data.shape[0]))

        # gloabl signals
        global_signal = data[np.where(atlas != 0)[0], :].mean(axis=0)
        global_diff = np.diff(global_signal, prepend=global_signal[0])

        # WM signals
        wm_ind = np.where(atlas >= 3000)[0]
        wm_mask = np.zeros(atlas.shape)
        wm_mask[wm_ind] = 1
        wm_mask = binary_erosion(wm_mask).reshape(atlas.shape)
        wm_signal = data[np.where(wm_mask == 1)[0], :].mean(axis=0)
        wm_diff = np.diff(wm_signal, prepend=wm_signal[0])

        # CSF signals
        csf_signal = data[[i for i in range(len(atlas)) if atlas[i] in csf_code]].mean(axis=0)
        csf_diff = np.diff(csf_signal, prepend=csf_signal[0])

        # We will not regress out motion parameters for FIX denoised data
        # see https://www.mail-archive.com/hcp-users@humanconnectome.org/msg02957.html
        # motion = pd.read_table(motion_file, sep='  ', header=None, engine='python')
        # motion = motion.join(np.power(motion, 2), lsuffix='motion', rsuffix='motion2')

        conf = np.vstack((global_signal, global_diff, wm_signal, wm_diff, csf_signal, csf_diff)).T
        return conf

    def _parcellate(self, t_surf: np.ndarray, t_vol: np.ndarray) -> dict:
        if self.inputs.config["dataset"] in ["HCP-YA", "HCP-A", "HCP-D"]:
            conf = self._nuisance_conf_hcp(t_vol)
        else:
            raise DatasetError()
        data_dir = files(mpp) / "data"

        regressors = np.concatenate((
            zscore(conf), np.ones((conf.shape[0], 1)),
            np.linspace(-1, 1, num=conf.shape[0]).reshape((conf.shape[0], 1))), axis=1)
        t_surf_resid = t_surf - np.dot(regressors, np.linalg.lstsq(regressors, t_surf, rcond=-1)[0])

        tavg_dict = {}
        for level in range(4):
            parc_sch_file = Path(
                data_dir, f"Schaefer2018_{level+1}00Parcels_17Networks_order.dlabel.nii")
            parc_mel_file = Path(data_dir, f"Tian_Subcortex_S{level+1}_3T.nii.gz")
            parc_sch = nib.load(parc_sch_file).get_fdata()
            parc_mel = nib.load(parc_mel_file).get_fdata()

            mask = parc_mel.nonzero()
            t_vol_subcort = np.array(
                [t_vol[mask[0][i], mask[1][i], mask[2][i], :] for i in range(mask[0].shape[0])])
            t_vol_resid = (
                    t_vol_subcort.T -
                    np.dot(regressors, np.linalg.lstsq(regressors, t_vol_subcort.T, rcond=-1)[0]))

            t_surf = t_surf_resid[:, range(parc_sch.shape[1])]
            parc_surf = np.zeros(((level + 1) * 100, t_surf.shape[0]))
            for parcel in range((level + 1) * 100):
                selected = t_surf[:, np.where(parc_sch == (parcel + 1))[1]]
                selected = selected[:, ~np.isnan(selected[0, :])]
                parc_surf[parcel, :] = selected.mean(axis=1)

            parcels = np.unique(parc_mel[mask]).astype(int)
            parc_vol = np.zeros((parcels.shape[0], t_vol_resid.shape[0]))
            for parcel in parcels:
                selected = t_vol_resid[:, np.where(parc_mel[mask] == parcel)[0]]
                selected = selected[:, ~np.isnan(selected[0, :])]
                selected = selected[
                           :, np.where(np.abs(selected.mean(axis=0)) >= sys.float_info.epsilon)[0]]
                parc_vol[parcel - 1, :] = selected.mean(axis=1)

            tavg_dict[f"level{level+1}"] = np.vstack((parc_surf, parc_vol))
        return tavg_dict

    def _sfc(self) -> dict:
        # static FC (Fisher's z excluding diagonals)
        sfc_dict = {}
        for level in range(4):
            fc = np.corrcoef(self._tavg_dict[f"level{level + 1}"])
            fc = (
                    0.5 * (np.log(1 + fc, where=~np.eye(fc.shape[0], dtype=bool)) -
                           np.log(1 - fc, where=~np.eye(fc.shape[0], dtype=bool))))
            fc[np.diag_indices_from(fc)] = 0
            sfc_dict[f"level{level + 1}"] = fc
        return sfc_dict

    def _dfc(self) -> None:
        # dynamic FC: 1st order ARR model
        # see https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/fMRI_dynamics/Liegeois2017_Surrogates/CBIG_RL2017_ar_mls.m # noqa: E501
        self._results["dfc"] = {}
        for level in range(4):
            tavg = self._tavg_dict[f"level{level+1}"]
            y = tavg[:, range(1, tavg.shape[1])]
            z = np.ones((tavg.shape[0] + 1, tavg.shape[1] - 1))
            z[1:(tavg.shape[0] + 1), :] = tavg[:, range(tavg.shape[1] - 1)]
            b = np.linalg.lstsq((z @ z.T).T, (y @ z.T).T, rcond=None)[0].T
            self._results["dfc"][f"level{level+1}"] = b[:, range(1, b.shape[1])]

    def _ev_block(self, length: int, ev_files: list) -> np.ndarray:
        task_reg = np.zeros((length, len(ev_files)))
        for ev_i, ev_file in enumerate(ev_files):
            ev = pd.read_table(ev_file, header=None, sep=" ")
            for _, block in ev.iterrows():
                block_start = int(
                    round(block[0] / self.inputs.config["param"]["tr"])) - 1
                block_len = int(round(block[1] / self.inputs.config["param"]["tr"]))
                task_reg[block_start:(block_start + block_len), ev_i] = 1
        return task_reg

    def _ec(self, task_reg: Optional[np.ndarray] = None) -> dict:
        # effective connectivity: regression DCM
        ec_dict = {}
        for level in range(4):
            rdcm = RegressionDCM(
                self._tavg_dict[f"level{level+1}"].T, self.inputs.config["param"]["tr"],
                task_reg, prior_a=self.inputs.sc_count[f"level{level+1}"])
            rdcm.estimate()
            ec_dict[f"level{level+1}"] = rdcm.params["mu_connectivity"]
        return ec_dict

    @staticmethod
    def _concat(tavg1: dict, tavg2: dict) -> dict:
        tavg = {}
        for key in tavg1.keys():
            tavg[key] = np.hstack((tavg1[key], tavg2[key]))
        return tavg

    def _run_interface(self, runtime):
        dataset = self.inputs.config["dataset"]
        if self.inputs.modality == "rfMRI":
            n_runs = len(self.inputs.rs_runs) + self.inputs.hcpd_b_runs
            self._tavg_dict = {}
            for i in range(n_runs):
                if dataset == "HCP-D" and i >= 4:
                    run = self.inputs.rs_runs[i-3]
                    key_surf = f"{run}_surfb"
                    key_vol = f"{run}_volb"
                else:
                    run = self.inputs.rs_runs[i]
                    key_surf = f"{run}_surf"
                    key_vol = f"{run}_vol"

                if self.inputs.data_files[key_surf] and self.inputs.data_files[key_vol]:
                    t_surf = nib.load(self.inputs.data_files[key_surf]).get_fdata()
                    t_vol = nib.load(self.inputs.data_files[key_vol]).get_fdata()
                    tavg = self._parcellate(t_surf, t_vol)
                    for key, val in tavg.items():
                        if key in self._tavg_dict.keys():
                            self._tavg_dict[key] = np.hstack((self._tavg_dict[key], val))
                        else:
                            self._tavg_dict[key] = val
            self._results["sfc"] = self._sfc()
            self._dfc()
            if self.inputs.sc_count:
                self._ec()

        elif self.inputs.modality == "tfMRI":
            self._results["sfc"] = {}
            for run in self.inputs.config["param"]["task_runs"]:
                if dataset == "HCP-YA":
                    self._tavg_dict = self._concat(
                        self._parcellate(
                            nib.load(self.inputs.data_files[f"{run}_LR_surf"]).get_fdata(),
                            nib.load(self.inputs.data_files[f"{run}_LR_vol"]).get_fdata()),
                        self._parcellate(
                            nib.load(self.inputs.data_files[f"{run}_RL_surf"]).get_fdata(),
                            nib.load(self.inputs.data_files[f"{run}_RL_vol"]).get_fdata()))
                elif dataset == "HCP-D":
                    if run == "tfMRI_EMOTION":
                        self._tavg_dict = self._parcellate(
                            nib.load(self.inputs.data_files[f"{run}_PA_surf"]).get_fdata(),
                            nib.load(self.inputs.data_files[f"{run}_PA_vol"]).get_fdata())
                    else:
                        self._tavg_dict = self._concat(
                            self._parcellate(
                                nib.load(self.inputs.data_files[f"{run}_PA_surf"]).get_fdata(),
                                nib.load(self.inputs.data_files[f"{run}_PA_vol"]).get_fdata()),
                            self._parcellate(
                                nib.load(self.inputs.data_files[f"{run}_AP_surf"]).get_fdata(),
                                nib.load(self.inputs.data_files[f"{run}_AP_vol"]).get_fdata()))
                elif dataset == "HCP-A":
                    self._tavg_dict = self._parcellate(
                        nib.load(self.inputs.data_files[f"{run}_surf"]).get_fdata(),
                        nib.load(self.inputs.data_files[f"{run}_vol"]).get_fdata())
                else:
                    raise DatasetError()
                self._results["sfc"][run] = self._sfc()

                if self.inputs.sc_count:
                    self._results["ec"] = {}
                    length_all = self._tavg_dict["level1"].shape[1]
                    if dataset == "HCP-YA":
                        ev_files_lr = self.inputs.data_files[f"{run}_LR_ev"]
                        task_reg_lr = self._ev_block(int(length_all/2), ev_files_lr)
                        ev_files_rl = self.inputs.data_files[f"{run}_RL_ev"]
                        task_reg_rl = self._ev_block(int(length_all/2), ev_files_rl)
                        task_reg = np.vstack((task_reg_lr, task_reg_rl))
                    elif dataset == "HCP-D" and run != "tfMRI_EMOTION":
                        ev_files_ap = self.inputs.data_files[f"{run}_AP_ev"]
                        task_reg_ap = self._ev_block(int(length_all/2), ev_files_ap)
                        ev_files_pa = self.inputs.data_files[f"{run}_PA_ev"]
                        task_reg_pa = self._ev_block(int(length_all/2), ev_files_pa)
                        task_reg = np.vstack((task_reg_ap, task_reg_pa))
                    else:
                        ev_files = self.inputs.data_files[f"{run}_ev"]
                        task_reg = self._ev_block(length_all, ev_files)
                    self._results["ec"][run] = self._ec(task_reg)

        return runtime


class _NetworkStatsInputSpec(BaseInterfaceInputSpec):
    conn = traits.Dict({}, usedefault=True, dtype=float, desc="connectivity matrix")


class _NetworkStatsOutputSpec(TraitedSpec):
    stats = traits.Dict(dtype=float, desc="network statistics features")


class NetworkStats(SimpleInterface):
    """Compute graph theory based network statistics from a connectivity matrix"""
    input_spec = _NetworkStatsInputSpec
    output_spec = _NetworkStatsOutputSpec

    def _run_interface(self, runtime):
        self._results["stats"] = {}
        for level in ["1", "2", "3", "4"]:
            stats_level = {}
            conn = self.inputs.conn[f"level{level}"]

            dist, _ = bct.distance_wei(bct.invert(conn))
            comm, _ = bct.community_louvain(conn, B="negative_sym")
            stats_level["cpl"], stats_level["eff"], _, _, _ = bct.charpath(dist)
            _, stats_level["mod"] = bct.modularity_und(conn, kci=comm)
            par = bct.participation_coef(conn, comm)
            for i in range(par.shape[0]):
                stats_level[f"par_{i}"] = par[i]
            self._results["stats"][f"level{level}"] = stats_level

        return runtime


class _AnatInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    data_files = traits.Dict(mandatory=True, dtype=Path, desc="collection of filenames")
    anat_dir = traits.Str(mandatory=True, desc="absolute path to installed subject T1w directory")
    conf = traits.Dict(mandatory=True, desc="confounding variables")
    simg_cmd = traits.Any(mandatory=True, desc="command for using singularity image (or not)")


class _AnatOutputSpec(TraitedSpec):
    myelin = traits.Dict(dtype=float, desc="myelin content estimates")
    morph = traits.Dict(dtype=float, desc="morphometry features")


class Anat(SimpleInterface):
    """Compute myelin estimate from T1dividedbyT2 files & morphometry features"""
    input_spec = _AnatInputSpec
    output_spec = _AnatOutputSpec

    def _run_interface(self, runtime):
        subject = self.inputs.config["subject"]
        data_dir = files(mpp) / "data"
        self._results["myelin"] = {}
        self._results["morph"] = {"gmv": {}, "cs": {}, "ct": {}}

        myelin_l = nib.load(self.inputs.data_files["myelin_l"]).agg_data()
        myelin_r = nib.load(self.inputs.data_files["myelin_r"]).agg_data()
        myelin_surf = np.hstack((myelin_l, myelin_r))
        myelin_vol = nib.load(self.inputs.data_files["myelin_vol"]).get_fdata()

        for level in range(4):
            # Myelin estimate
            parc_sch_file = Path(
                data_dir, f"Schaefer2018_{level+1}00Parcels_17Networks_order.dlabel.nii")
            parc_mel_file = Path(data_dir, f"Tian_Subcortex_S{level+1}_3T.nii.gz")
            parc_sch = nib.load(parc_sch_file).get_fdata()
            parc_mel = nib.load(parc_mel_file).get_fdata()
            parc_surf = np.zeros(((level+1)*100))
            for parcel in range((level+1)*100):
                selected = myelin_surf[np.where(parc_sch == (parcel + 1))[1]]
                selected = selected[~np.isnan(selected)]
                parc_surf[parcel] = selected.mean()
            parc_mel_mask = parc_mel.nonzero()
            parc_mel = parc_mel[parc_mel_mask]
            myelin_vol_masked = np.array([
                myelin_vol[parc_mel_mask[0][i], parc_mel_mask[1][i], parc_mel_mask[2][i]]
                for i in range(parc_mel_mask[0].shape[0])])
            parcels = np.unique(parc_mel).astype(int)
            parc_vol = np.zeros((parcels.shape[0]))
            for parcel in parcels:
                selected = myelin_vol_masked[np.where(parc_mel == parcel)[0]]
                selected = selected[~np.isnan(selected)]
                parc_vol[parcel-1] = selected.mean()
            self._results["myelin"][f"level{level+1}"] = np.hstack([parc_surf, parc_vol])

            # CS & CT
            stats_surf = None
            for hemi in ["lh", "rh"]:
                annot_file = f"{hemi}.Schaefer2018_{level+1}00Parcels_17Networks_order.annot"
                annot_fs = files(mpp) / "data" / annot_file
                annot_sub = Path(self.inputs.config["tmp_dir"], f"{hemi}.{subject}_{level+1}.annot")
                add_subdir = AddSubDir(
                    sub_dir=self.inputs.config["tmp_dir"], subject=subject,
                    fs_dir=Path(self.inputs.anat_dir, subject))
                add_subdir.run()
                fs_opts = f"--env SUBJECTS_DIR={self.inputs.config['tmp_dir']}"
                annot_fs2sub = freesurfer.SurfaceTransform(
                    command=self.inputs.simg_cmd.cmd("mri_surf2surf", options=fs_opts),
                    source_annot_file=annot_fs, out_file=annot_sub, hemi=hemi,
                    source_subject="fsaverage", target_subject=subject,
                    subjects_dir=self.inputs.config["tmp_dir"])
                annot_fs2sub.run()
                hemi_table = Path(self.inputs.config["tmp_dir"], f"{hemi}.{subject}_fs_stats")
                options = f"--env SUBJECTS_DIR={self.inputs.anat_dir}"
                subprocess.run(
                    self.inputs.simg_cmd.cmd("mris_anatomical_stats", options=options).split() + [
                        "-a", str(annot_sub), "-noglobal", "-f", str(hemi_table), subject, hemi],
                    env=dict(environ, **{"SUBJECTS_DIR": self.inputs.anat_dir}), check=True)
                hemi_stats = pd.read_table(
                    hemi_table, header=0, skiprows=np.arange(51), delim_whitespace=True)
                hemi_stats.drop([0], inplace=True)  # exclude medial wall
                if stats_surf is None:
                    stats_surf = hemi_stats
                else:
                    stats_surf = pd.concat([stats_surf, hemi_stats])
            # Divide CS by ICV^(2/3)
            self._results["morph"]["cs"][f"level{level+1}"] = np.divide(
                stats_surf["SurfArea"].values, np.power(self.inputs.conf["icv_vol"], 2/3))
            # Divide CT by mean CT
            self._results["morph"]["ct"][f"level{level+1}"] = np.divide(
                stats_surf["ThickAvg"].values, stats_surf["ThickAvg"].mean())

            # GMV
            seg_up_file = Path(self.inputs.config["tmp_dir"], f"{subject}_S{level}_up.nii.gz")
            flt = fsl.FLIRT(
                command=self.inputs.simg_cmd.cmd(cmd="flirt"), in_file=parc_mel_file,
                reference=self.inputs.data_files["t1_vol"], out_file=seg_up_file,
                apply_isoxfm=self.inputs.config["param"]["t1_res"], interp="nearestneighbour")
            flt.run()
            sub_table = "subcortex.stats"
            ss = freesurfer.SegStats(
                command=self.inputs.simg_cmd.cmd(
                    "mri_segstats", options=f"--env SUBJECTS_DIR={self.inputs.anat_dir}"),
                segmentation_file=seg_up_file, in_file=self.inputs.data_files['t1_vol'],
                summary_file=sub_table, subjects_dir=self.inputs.anat_dir)
            ss.run()
            stats_vol = pd.read_table(
                sub_table, header=0, skiprows=np.arange(50), delim_whitespace=True)
            stats_vol.drop([0], inplace=True)
            # Divide GMV by ICV
            self._results["morph"]["gmv"][f"level{level+1}"] = np.divide(
                np.concatenate((stats_surf["GrayVol"].values, stats_vol["Volume_mm3"].values)),
                self.inputs.conf["icv_vol"])

        return runtime


class _SCInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    atlas_files = traits.List(mandatory=True, dtype=Path, desc="Atlas files")
    tck_file = traits.File(mandatory=True, exists=True, desc="Tractogram file")
    fs_dir = traits.Directory(desc="FreeSurfer subject directory")
    simg_cmd = traits.Any(mandatory=True, desc="command for using singularity image (or not)")


class _SCOutputSpec(TraitedSpec):
    sc_count = traits.Dict(desc="structural connectome based on streamline counts")
    sc_length = traits.Dict(desc="structural connectome based on streamline length")


class SC(SimpleInterface):
    """Compute structural connectivity based on streamline count and length-scaled count"""
    input_spec = _SCInputSpec
    output_spec = _SCOutputSpec

    def _run_interface(self, runtime):
        work_dir = self.inputs.config["tmp_dir"]
        self._results["sc_count"] = {}
        self._results["sc_length"] = {}

        for atlas_file in self.inputs.atlas_files:
            key = f"level{Path(atlas_file).name.split('flirt')[0][-2]}"
            count_file = Path(work_dir, f"sc_count_{key}.csv")
            subprocess.run(
                self.inputs.simg_cmd.cmd("tck2connectome").split() + [
                    "-assignment_radial_search", "2", "-symmetric", "-nthreads", "0", "-force",
                    str(self.inputs.tck_file), str(atlas_file), str(count_file)],
                check=True)
            self._results["sc_count"][key] = np.log(pd.read_csv(count_file, header=None))

            length_file = Path(work_dir, f"sc_length_{key}.csv")
            subprocess.run(
                self.inputs.simg_cmd.cmd("tck2connectome").split() + [
                    "-assignment_radial_search", "2", "-scale_length", "-stat_edge", "mean",
                    "-symmetric", "-nthreads", "0", "-force", str(self.inputs.tck_file),
                    str(atlas_file), str(length_file)],
                check=True)
            self._results["sc_length"][key] = pd.read_csv(length_file, header=None)

        return runtime


class _DTIFeaturesInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    fa_skeleton_file = traits.File(mandatory=True, exists=True, desc="skeletonised FA file")
    md_skeleton_file = traits.File(mandatory=True, exists=True, desc="skeletonised MD file")
    ad_skeleton_file = traits.File(mandatory=True, exists=True, desc="skeletonised AD file")
    rd_skeleton_file = traits.File(mandatory=True, exists=True, desc="skeletonised RD file")
    sublist = traits.List(mandatory=True, dtype=str, desc="subject IDs")


class _DTIFeaturesOutputSpec(TraitedSpec):
    dti_features = traits.Dict(desc="region-wise DTI features")


class DTIFeatures(SimpleInterface):
    """Extract DTI features"""
    input_spec = _DTIFeaturesInputSpec
    output_spec = _DTIFeaturesOutputSpec

    def _run_interface(self, runtime):
        in_files = {
            "fa": self.inputs.fa_skeleton_file, "md": self.inputs.md_skeleton_file,
            "ad": self.inputs.ad_skeleton_file, "rd": self.inputs.rd_skeleton_file}
        parc_jhu_file = files(mpp) / "data" / "JHU-ICBM-labels-1mm.nii.gz"
        parc_jhu = nib.load(parc_jhu_file).get_fdata()
        parc_jhu_mask = parc_jhu.nonzero()
        parc_jhu = parc_jhu[parc_jhu_mask]
        parcels = np.unique(parc_jhu).astype(int)

        self._results["dti_features"] = {"fa": {}, "md": {}, "ad": {}, "rd": {}}
        for file_type, file_in in in_files.items():
            data = nib.load(file_in).get_fdata()
            for sub_i, subject in enumerate(self.inputs.sublist):
                data_masked = np.array([
                    data[parc_jhu_mask[0][i], parc_jhu_mask[1][i], parc_jhu_mask[2][i], sub_i]
                    for i in range(parc_jhu_mask[0].shape[0])])
                data_parc = np.zeros((parcels.shape[0]))
                for parcel in parcels:
                    selected = data_masked[np.where(parc_jhu == parcel)[0]]
                    selected = selected[~np.isnan(selected)]
                    data_parc[parcel-1] = selected.mean()
                self._results["dti_features"][file_type][subject] = data_parc

        return runtime


class _ConfoundsInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    data_files = traits.Dict(mandatory=True, dtype=Path, desc="collection of filenames")
    simg_cmd = traits.Any(mandatory=True, desc="command for using singularity image (or not)")


class _ConfoundsOutputSpec(TraitedSpec):
    conf = traits.Dict(desc="confounding variables")


class Confounds(SimpleInterface):
    """Extract confounding variables"""
    input_spec = _ConfoundsInputSpec
    output_spec = _ConfoundsOutputSpec

    def _run_interface(self, runtime):
        # primary variables
        if self.inputs.config["dataset"] == "HCP-YA":
            unres_conf = pd.read_csv(
                self.inputs.config["param"]["pheno_file"],
                usecols=["Subject", "Gender", "FS_BrainSeg_Vol", "FS_IntraCranial_Vol"],
                dtype={"Subject": str, "Gender": str, "FS_BrainSeg_Vol": float,
                       "FS_IntraCranial_Vol": float})
            unres_conf = unres_conf.loc[unres_conf["Subject"] == self.inputs.config["subject"]]
            res_conf = pd.read_csv(
                self.inputs.config["param"]["restricted_file"],
                usecols=["Subject", "Age_in_Yrs", "Handedness"],
                dtype={"Subject": str, "Age_in_Yrs": int, "Handedness": int})
            res_conf = res_conf.loc[res_conf["Subject"] == self.inputs.config["subject"]]
            conf = {
                "age": res_conf["Age_in_Yrs"].values[0], "gender": unres_conf["Gender"].values[0],
                "handedness": res_conf["Handedness"].values[0],
                "brainseg_vol": unres_conf["FS_BrainSeg_Vol"].values[0],
                "icv_vol": unres_conf["FS_IntraCranial_Vol"].values[0]}
        elif self.inputs.config["dataset"] in ["HCP-A", "HCP-D"]:
            subject = self.inputs.config["subject"].split("_V1_MR")[0]
            demo = pd.read_table(
                self.inputs.config["param"]["demo_file"], sep="\t", header=0, skiprows=[1],
                usecols=[4, 5, 7], dtype={"src_subject_id": str, "interview_age": int, "sex": str})
            demo = demo.loc[demo["src_subject_id"] == subject]
            handedness = pd.read_table(
                self.inputs.config["param"]["hand_file"], sep="\t", header=0, skiprows=[1],
                usecols=[5, 70], dtype={"src_subject_id": str, "hcp_handedness_score": int})
            handedness = handedness.loc[handedness["src_subject_id"] == subject]
            conf = {
                "age": demo["interview_age"].values[0], "gender": demo["sex"].values[0],
                "handedness": handedness["hcp_handedness_score"].values[0]}

            astats_file = Path(self.inputs.config["tmp_dir"], f"{subject}_astats.txt")
            subprocess.run(
                self.inputs.simg_cmd.cmd("asegstats2table").split()
                + ["--meas", "volume", "--tablefile", str(astats_file)]
                + ["--inputs", str(self.inputs.data_files["astats"])], check=True)
            aseg_stats = pd.read_csv(str(astats_file), sep="\t", index_col=0)
            conf["brainseg_vol"] = aseg_stats["BrainSegVol"].values[0]
            conf["icv_vol"] = aseg_stats["EstimatedTotalIntraCranialVol"].values[0]
        else:
            raise DatasetError()

        # gender coding: 1 for Female, 2 for Male
        conf["gender"] = 1 if conf["gender"] == "F" else 2
        # secondary variables
        conf["age2"] = np.power(conf["age"], 2)
        conf["ageGender"] = conf["age"] * conf["gender"]
        conf["age2Gender"] = conf["age2"] * conf["gender"]

        self._results["conf"] = conf

        return runtime
