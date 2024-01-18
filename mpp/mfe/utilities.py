from os import getenv, symlink
from pathlib import Path
from shutil import copytree
from typing import Union

from nipype.interfaces.base import BaseInterfaceInputSpec, TraitedSpec, SimpleInterface, traits
import nibabel as nib
import numpy as np

from mpp.exceptions import DatasetError

base_dir = Path(__file__).resolve().parent.parent.parent


class SimgCmd:
    def __init__(self, config: dict) -> None:
        if config["simg"] is None:
            self._cmd = None
        else:
            self._cmd = (
                f"singularity run -B {config['work_dir']}:{config['work_dir']},"
                f"{config['output_dir']}:{config['output_dir']},{base_dir}:{base_dir}")
            self._simg = config["simg"]

    def cmd(self, cmd: str, options: Union[str, None] = None) -> str:
        if self._cmd is None:
            return cmd
        else:
            if options is None:
                return f"{self._cmd} {self._simg} {cmd}"
            else:
                return f"{self._cmd} {options} {self._simg} {cmd}"


def dataset_params(dataset: str, root_data_dir: Path, pheno_dir: Path, subject: str) -> dict:
    params = {
        "HCP-YA": {
            "url": "git@github.com:datalad-datasets/human-connectome-project-openaccess.git",
            "source": None,
            "dir": Path(root_data_dir),
            "sub_dir": Path(root_data_dir, "HCP1200", subject),
            "clean": "hp2000_clean",
            "tasks": [
                "tfMRI_EMOTION_LR", "tfMRI_EMOTION_RL", "tfMRI_GAMBLING_LR", "tfMRI_GAMBLING_RL",
                "tfMRI_LANGUAGE_LR", "tfMRI_LANGUAGE_RL", "tfMRI_MOTOR_LR", "tfMRI_MOTOR_RL",
                "tfMRI_RELATIONAL_LR", "tfMRI_RELATIONAL_RL", "tfMRI_SOCIAL_LR", "tfMRI_SOCIAL_RL",
                "tfMRI_WM_LR", "tfMRI_WM_RL"],
            "rests": ['rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR', 'rfMRI_REST2_RL'],
            "task_runs": [
                "tfMRI_EMOTION", "tfMRI_GAMBLING", "tfMRI_LANGUAGE", "tfMRI_MOTOR", "tfMRI_WM",
                "tfMRI_RELATIONAL", "tfMRI_SOCIAL"],
            "t1_res": 0.7,
            "shells": "1000,2000,3000",
            "diff_res": 1.25,
            "col_names": {
                "totalcogcomp": "CogTotalComp_AgeAdj", "fluidcogcomp": "CogFluidComp_AgeAdj",
                "crycogcomp": "CogCrystalComp_AgeAdj", "cardsort": "CardSort_AgeAdj",
                "flanker": "Flanker_AgeAdj", "reading": "ReadEng_AgeAdj",
                "picvocab": "PicVocab_AgeAdj", "procspeed": "ProcSpeed_AgeAdj",
                "ddisc": "DDisc_AUC_40K", "listsort": "ListSort_AgeAdj", "emotrecog": "ER40_CR",
                "anger": "AngAffect_Unadj", "fear": "FearAffect_Unadj", "sadness": "Sadness_Unadj",
                "posaffect": "PosAffect_Unadj", "emotsupp": "EmotSupp_Unadj",
                "friendship": "Friendship_Unadj", "loneliness": "Loneliness_Unadj",
                "endurance": "Endurance_AgeAdj", "gaitspeed": "GaitSpeed_Comp",
                "strength": "Strength_AgeAdj", "neoffi_n": "NEOFAC_N", "neoffi_e": "NEOFAC_E",
                "neoffi_o": "NEOFAC_O", "neoffi_a": "NEOFAC_A", "neoffi_c": "NEOFAC_C"},
            "pheno_file": Path(pheno_dir, "unrestricted_hcpya.csv"),
            "restricted_file": Path(pheno_dir, "restricted_hcpya.csv"),
            "tr": 0.72,
            "ev_files": {
                "tfMRI_EMOTION": ["fear.txt", "neut.txt"],
                "tfMRI_GAMBLING": ["win_event.txt", "loss_event.txt", "neut_event.txt"],
                "tfMRI_LANGUAGE": ["story.txt", "math.txt"],
                "tfMRI_MOTOR": ["cue.txt", "lf.txt", "rf.txt", "lh.txt", "rh.txt", "t.txt"],
                "tfMRI_WM": [
                    "0bk_body.txt", "0bk_faces.txt", "0bk_places.txt", "0bk_tools.txt",
                    "2bk_body.txt", "2bk_faces.txt", "2bk_places.txt", "2bk_tools.txt"],
                "tfMRI_RELATIONAL": ["relation.txt", "match.txt"],
                "tfMRI_SOCIAL": ["mental_resp.txt", "other_resp.txt"]}},
        "HCP-A": {
            "url": "git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git",
            "source": "inm7-storage",
            "dir": Path(root_data_dir, "original", "hcp", "hcp_aging"),
            "sub_dir": Path(root_data_dir, "original", "hcp", "hcp_aging", subject),
            "diff_url": "git@gin.g-node.org:/jadecci/hcp_lifespan_diffproc.git",
            "clean": "hp0_clean",
            "tasks": ["tfMRI_CARIT_PA", "tfMRI_FACENAME_PA", "tfMRI_VISMOTOR_PA"],
            "rests": ['rfMRI_REST1_AP', 'rfMRI_REST1_PA', 'rfMRI_REST2_AP', 'rfMRI_REST2_PA'],
            "task_runs": ["tfMRI_CARIT_PA", "tfMRI_FACENAME_PA", "tfMRI_VISMOTOR_PA"],
            "t1_res": 0.8,
            "shells": "1500,3000",
            "diff_res": 1.5,
            "col_names": {
                "totalcogcomp": "nih_totalcogcomp_ageadjusted",
                "fluidcogcomp": "nih_fluidcogcomp_ageadjusted",
                "crycogcomp": "nih_crycogcomp_ageadjusted", "cardsort": "nih_dccs_ageadjusted",
                "flanker": "nih_flanker_ageadjusted", "reading": "read_acss",
                "picvocab": "tpvt_acss", "procspeed": "nih_patterncomp_ageadjusted",
                "ddisc": "auc_40000", "listsort": "age_corrected_standard_score",
                "emotrecog": "er40_c_cr", "anger": "anger_ts", "fear": "anx_ts",
                "sadness": "add_ts", "posaffect": "tlbxpa_ts", "emotsupp": "nih_tlbx_tscore",
                "friendship": "nih_tlbx_tscore", "loneliness": "soil_ts",
                "endurance": "end_2m_standardsc", "gaitspeed": "loco_comscore",
                "strength": "grip_standardsc_dom", "neoffi_n": "neo2_score_ne",
                "neoffi_e": "neo2_score_ex", "neoffi_o": "neo2_score_op",
                "neoffi_a": "neo2_score_ag", "neoffi_c": "neo2_score_co"},
            "pheno_files": {
                "totalcogcomp": "cogcomp01.txt", "fluidcogcomp": "cogcomp01.txt",
                "crycogcomp": "cogcomp01.txt", "cardsort": "dccs01.txt", "flanker": "flanker01.txt",
                "reading": "orrt01.txt", "picvocab": "tpvt01.txt", "procspeed": "pcps01.txt",
                "ddisc": "deldisk01.txt", "listsort": "lswmt01.txt", "emotrecog": "er4001.txt",
                "anger": "prang01.txt", "fear": "preda01.txt", "sadness": "predd01.txt",
                "posaffect": "tlbx_wellbeing01.txt", "emotsupp": "tlbx_emsup01.txt",
                "friendship": "tlbx_friend01.txt", "loneliness": "prsi01.txt",
                "endurance": "tlbx_motor01.txt", "gaitspeed": "tlbx_motor01.txt",
                "strength": "tlbx_motor01.txt", "neoffi_n": "nffi01.txt", "neoffi_e": "nffi01.txt",
                "neoffi_o": "nffi01.txt", 'neoffi_a': "nffi01.txt", 'neoffi_c': "nffi01.txt"},
            "pheno_cols": {
                "totalcogcomp": 30, "fluidcogcomp": 14, "crycogcomp": 18, "cardsort": 76,
                "flanker": 55, "reading": 10, "picvocab": 10, "procspeed": 145, "ddisc": 133,
                "listsort": 136, "emotrecog": 8, "anger": 31, "fear": 38, "sadness": 37,
                "posaffect": 157, "emotsupp": 12, "friendship": 12, "loneliness": 23,
                "endurance": 16, "gaitspeed": 30, "strength": 22, "neoffi_n": 77,
                "neoffi_e": 76, "neoffi_o": 78, "neoffi_a": 74, "neoffi_c": 75},
            "demo_file": Path(pheno_dir, "ssaga_cover_demo01.txt"),
            "hand_file": Path(pheno_dir, "edinburgh_hand01.txt"),
            "tr": 0.8,
            "ev_files": {
                "tfMRI_CARIT_PA":["go.txt", "miss.txt", "nogoCR.txt", "nogoFA.txt"],
                "tfMRI_FACENAME_PA": ["encoding.txt", "recall.txt"],
                "tfMRI_VISMOTOR_PA": ["vismotor.txt"]}},
        "HCP-D": {
            "url": "git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git",
            "source": "inm7-storage",
            "dir": Path(root_data_dir, "original", "hcp", "hcp_development"),
            "sub_dir": Path(root_data_dir, "original", "hcp", "hcp_development", subject),
            "diff_url": "git@gin.g-node.org:/jadecci/hcp_lifespan_diffproc.git",
            "clean": "hp0_clean",
            "tasks": [
                "tfMRI_CARIT_AP", "tfMRI_CARIT_PA", "tfMRI_EMOTION_PA", "tfMRI_GUESSING_AP",
                "tfMRI_GUESSING_PA"],
            "rests": ['rfMRI_REST1_AP', 'rfMRI_REST1_PA', 'rfMRI_REST2_AP', 'rfMRI_REST2_PA'],
            "task_runs": ["tfMRI_CARIT", "tfMRI_EMOTION", "tfMRI_GUESSING"],
            "t1_res": 0.8,
            "shells": "1500,3000",
            "diff_res": 1.5,
            "pheno_cols": {
                "totalcogcomp": 18, "fluidcogcomp": 9, "crycogcomp": 12, "cardsort": 41,
                "flanker": 11, "reading": 10, "picvocab": 10, "procspeed": 10, "ddisc": 22,
                "listsort": 36, "emotrecog": 8, "anger": 29, "fear": 28, "sadness": 26,
                "posaffect": 66, "emotsupp": 11, "friendship": 10, "loneliness": 11,
                "endurance": 13, "gaitspeed": 18, "strength": 14, "neoffi_n": 71,
                "neoffi_e": 70, "neoffi_o": 72, "neoffi_a": 68, "neoffi_c": 69},
            "demo_file": Path(pheno_dir, "ssaga_cover_demo01.txt"),
            "hand_file": Path(pheno_dir, "edinburgh_hand01.txt"),
            "tr": 0.8,
            "ev_files": {
                "tfMRI_CARIT": [
                    "go.txt", "miss.txt", "nogoCRLose.txt", "nogoCRWin.txt", "nogoFALose.txt",
                    "nogoFAWin.txt"],
                "tfMRI_EMOTION": ["faces.txt", "shapes.txt"],
                "tfMRI_GUESSING": [
                    "cueHigh.txt", "cueLow.txt", "feedbackHighLose.txt", "feedbackHighWin.txt",
                    "feedbackLowLose.txt", "feedbackLowWin.txt", "guess.txt"]}}}
    params["HCP-D"]["col_names"] = params["HCP-A"]["col_names"]
    params["HCP-D"]["pheno_files"] = params["HCP-A"]["pheno_files"]

    if dataset not in ["HCP-YA", "HCP-A", "HCP-D"]:
        raise DatasetError
    else:
        return params[dataset]


class _AddSubDirInputSpec(BaseInterfaceInputSpec):
    sub_dir = traits.Directory(mandatory=True, desc="Subject directory to create")
    subject = traits.Str(mandatory=True, desc="Subject ID")
    fs_dir = traits.Directory(mandatory=True, desc="Subject FreeSurfer ouptut directory")


class _AddSubDirOutputSpec(TraitedSpec):
    sub_dir = traits.Directory(exists=True, desc="Subject directory to create")


class AddSubDir(SimpleInterface):
    """Create a subject directory with subject and fsaverage data"""
    input_spec = _AddSubDirInputSpec
    output_spec = _AddSubDirOutputSpec

    def _run_interface(self, runtime):
        Path(self.inputs.sub_dir).mkdir(parents=True, exist_ok=True)
        if not Path(self.inputs.sub_dir, self.inputs.subject).is_symlink():
            symlink(self.inputs.fs_dir, Path(self.inputs.sub_dir, self.inputs.subject))
        if not Path(self.inputs.sub_dir, "fsaverage").is_dir():
            copytree(
                Path(getenv("FREESURFER_HOME"), "subjects", "fsaverage"),
                Path(self.inputs.sub_dir, "fsaverage"), dirs_exist_ok=True)
        self._results["sub_dir"] = self.inputs.sub_dir
        return runtime


class _CombineStringsInputSpec(BaseInterfaceInputSpec):
    input1 = traits.Str(mandatory=True, desc="Input string 1")
    input2 = traits.Str(mandatory=True, desc="Input string 2")
    input3 = traits.Str("", desc="Input string 3")
    input4 = traits.Str("", desc="Input string 4")


class _StringOutputSpec(TraitedSpec):
    output = traits.Str(desc="Output string")


class CombineStrings(SimpleInterface):
    """Combine multile strings into one"""
    input_spec = _CombineStringsInputSpec
    output_spec = _StringOutputSpec

    def _run_interface(self, runtime):
        self._results["output"] = f"{self.inputs.input1}{self.inputs.input2}" \
                                  f"{self.inputs.input3}{self.inputs.input4}"
        return runtime


class _CombineAtlasInputSpec(BaseInterfaceInputSpec):
    config = traits.Dict(mandatory=True, desc="Workflow configurations")
    cort_file = traits.File(mandatory=True, exists=True, desc='cortex atlas in T1 space')
    subcort_file = traits.File(mandatory=True, exists=True, desc='subcortex atlas in t1 space')
    level = traits.Str(mandatory=True, desc='parcellation level (0 to 3)')


class _CombineAtlasOutputSpec(TraitedSpec):
    combined_file = traits.File(exists=True, desc='combined atlas in T1 space')


class CombineAtlas(SimpleInterface):
    """combine cortex and subcortex atlases in T1 space"""
    input_spec = _CombineAtlasInputSpec
    output_spec = _CombineAtlasOutputSpec

    def _run_interface(self, runtime):
        atlas_img = nib.load(self.inputs.cort_file)
        atlas_cort = atlas_img.get_fdata()
        atlas_subcort = nib.load(self.inputs.subcort_file).get_fdata()
        atlas = np.zeros(atlas_cort.shape)

        cort_parcels = np.unique(atlas_cort[np.where(atlas_cort > 1000)])
        for parcel in cort_parcels:
            if 1000 < parcel < 2000:  # lh
                atlas[atlas_cort == parcel] = parcel - 1000
            elif parcel > 2000:  # rh
                atlas[atlas_cort == parcel] = parcel - 2000 + (len(cort_parcels) - 1) / 2

        for parcel in np.unique(atlas_subcort):
            if parcel != 0:
                atlas[atlas_subcort == parcel] = parcel + 100 * int(self.inputs.level)

        self._results["combined_file"] = Path(
            self.inputs.config["work_dir"], f"atlas_combine_level{self.inputs.level}.nii.gz")
        nib.save(
            nib.Nifti1Image(atlas, header=atlas_img.header, affine=atlas_img.affine),
            self._results["combined_file"])

        return runtime
