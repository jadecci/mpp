from pathlib import Path
from typing import Union

from mpp.exceptions import DatasetError


def d_files_dirsphase(d_files: dict, dirs: str, phase: str) -> tuple[str, ...]:
    key = f'dir{dirs}_{phase}'
    image = d_files[f'{key}.nii.gz']
    bval = d_files[f'{key}.bval']
    bvec = d_files[f'{key}.bvec']

    return str(image), str(bval), str(bvec)


def d_files_type(d_files: dict) -> tuple[list, ...]:
    image = [d_files[key] for key in d_files if '.nii.gz' in key]
    bval = [d_files[key] for key in d_files if '.bval' in key]
    bvec = [d_files[key] for key in d_files if '.bvec' in key]

    return image, bval, bvec


def t1_files_type(t1_files: dict) -> tuple[str, ...]:
    t1_file = t1_files['t1']
    t1_restore_file = t1_files['t1_restore']
    t1_brain_file = t1_files['t1_restore_brain']
    bias_file = t1_files['bias']
    mask_file = t1_files['fs_mask']
    t1_to_mni = t1_files['t1_to_mni']

    return (
        str(t1_file), str(t1_restore_file), str(t1_brain_file), str(bias_file), str(mask_file),
        str(t1_to_mni))


def fs_files_type(fs_files: dict) -> tuple[str, ...]:

    lh_whitedeform_file = fs_files['lh_white_deformed']
    rh_whitedeform_file = fs_files['rh_white_deformed']
    eye_file = fs_files['eye']
    orig_file = fs_files['orig']
    lh_thick_file = fs_files['lh_thickness']
    rh_thick_file = fs_files['rh_thickness']

    return (
        str(lh_whitedeform_file), str(rh_whitedeform_file), str(eye_file), str(orig_file),
        str(lh_thick_file), str(rh_thick_file))


def fs_files_aparc(fs_files: dict) -> tuple[str, ...]:
    lh_aparc = fs_files['lh_aparc']
    rh_aparc = fs_files['rh_aparc']
    lh_white = fs_files['lh_white']
    rh_white = fs_files['rh_white']
    lh_pial = fs_files['lh_pial']
    rh_pial = fs_files['rh_pial']
    lh_ribbon = fs_files['lh_ribbon']
    rh_ribbon = fs_files['rh_ribbon']
    ribbon = fs_files['ribbon']

    return (
        str(lh_aparc), str(rh_aparc), str(lh_white), str(rh_white), str(lh_pial), str(rh_pial),
        str(lh_ribbon), str(rh_ribbon), str(ribbon))


def update_d_files(d_files: dict, dataset: str, dwi_replacements: list) -> dict:
    if dataset == 'HCP-A' or dataset == 'HCP-D':
        keys = ['dir98_AP', 'dir98_PA', 'dir99_AP', 'dir99_PA']
    else:
        raise DatasetError()

    for key in keys:
        dwi_key = [d_key for d_key in d_files if key in d_key and '.nii.gz' in d_key]
        dwi_replace = [d_file for d_file in dwi_replacements if key in str(d_file)]
        d_files[dwi_key[0]] = dwi_replace[0]

    return d_files


def flatten_list(in_list: list) -> list:
    import itertools
    return list(itertools.chain.from_iterable(in_list))


def create_2item_list(item1: str, item2: str) -> list:
    return [item1, item2]


def combine_2strings(str1: str, str2: str) -> str:
    return f'{str1}{str2}'


def combine_4strings(str1: str, str2: str, str3: str, str4: str) -> str:
    return f'{str1}{str2}{str3}{str4}'


def last_list_item(in_list: list) -> str:
    return str(in_list[-1])


def diff_res(data_file: str) -> tuple[float, int]:
    import nibabel as nib
    res = nib.load(data_file).header.get_zooms()[0]

    return res, int(res*4)


def flirt_bbr_sch(
        in_file: str, reference: str, wm_seg: str, in_matrix_file: str, run_cmd: str,
        sch_file: str, out_dir: str, out_prefix: str) -> str:
    import subprocess
    from pathlib import Path

    out_matrix_file = Path(out_dir, f'flirt_sch_{out_prefix}.mat')
    subprocess.run(
        run_cmd.split() + [
            '-in', in_file, '-ref', reference, '-wmseg', wm_seg,
            '-init', in_matrix_file, '-omat', out_matrix_file, '-dof', '6',
            '-cost', 'bbr', '-schedule', sch_file],
        check=True
    )

    return str(out_matrix_file)


def bet_nodif_mask(in_file: str, run_cmd: str, work_dir: str) -> str:
    import subprocess
    from pathlib import Path

    out_file = Path(work_dir, f'{Path(in_file).stem}.nii.gz')
    mask_file = Path(work_dir, f'{Path(in_file).stem}_mask.nii.gz')
    subprocess.run(run_cmd.split() + [in_file, out_file, '-f', '0.20', '-m'], check=True)

    return str(mask_file)

