import pandas as pd
import datalad.api as dl
from os import path, listdir
from scipy.io import loadmat
from sys import stdout
import argparse

parser = argparse.ArgumentParser(
    description='generate a sublist of ABCD subjects with RS data and which passed QC',
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument(
    'dataset_dir', type=str, help='path to existing installation of ABCD dataset from '
                                  'https://collection3165.readthedocs.io/')
parser.add_argument('output', type=str, help='name for output sublist file')
parser.add_argument(
    '--source', dest='source', type=str, default=None, help='source to use for datalad get')
parser.add_argument(
    '--log', dest='log', type=str, default=None, help='log file to write exclusion causes to')
args = parser.parse_args()

# log file
if args.log is None:
    log = stdout
else:
    log = open(args.log, 'w')

# dataset-wise scanner info
pheno_dir = path.join(args.dataset_dir, 'phenotype')
scanner_info_file = path.join(pheno_dir, 'phenotype', 'abcd_mri01.txt')
dl.get(path=pheno_dir, dataset=args.dataset_dir, get_data=False)
dl.get(path=scanner_info_file, dataset=pheno_dir)
scanner_info = pd.read_table(
    scanner_info_file, delim_whitespace=True, usecols=['subjectkey', 'mri_info_manufacturer'],
    dtype=str)
dl.drop(scanner_info_file, reckless='kill')

# all subjects with derivatives data
deriv_dir = path.join(args.dataset_dir, 'derivatives', 'abcd-hcp-pipeline')
dl.get(path=deriv_dir, dataset=args.dataset_dir, get_data=False)
sublist_start = [
    f'NDAR_{item.lstrip("sub-NDAR")}' for item in listdir(deriv_dir) if 'sub-NDARINV' in item]
sublist = []
for subject in sublist_start:

    # exclude subject if Philips scanner was used
    scanners = scanner_info['mri_info_manufacturer'].loc[scanner_info['subjectkey'] == subject]
    if any('Philips' in scanner for scanner in scanners.dropna()):
        print(f'{subject} used Philips scanner', file=log)
        continue

    # check censored frames for each resting-state run
    sub_dir = path.join(deriv_dir, f'sub-NDAR{subject.lstrip("NDAR_")}')
    dl.get(path=sub_dir, dataset=deriv_dir, get_data=False)
    func_dir = path.join(sub_dir, 'ses-baselineYear1Arm1', 'func')
    censor_file = (f'sub-NDAR{subject.lstrip("NDAR_")}_ses-baselineYear1Arm1_task-rest_desc-'
                   f'filtered_motion_mask.mat')

    if path.islink(path.join(func_dir, censor_file)):
        dl.get(path=path.join(func_dir, censor_file), dataset=sub_dir, source=args.source)
        motion_data = loadmat(path.join(func_dir, censor_file))['motion_data'][0]
        total_frames = motion_data[30][0][0][5][0][0]
        removed_frames = motion_data[30][0][0][3]
        remain_seconds = motion_data[30][0][0][7][0][0]
        dl.drop(path.join(func_dir, censor_file), reckless='kill')

        # exclude subject if less than 4 min of data remained after censoring
        if remain_seconds/60 < 4:
            print(
                f'{subject} has {remain_seconds/60} minutes of RS data left after censoring',
                file=log)
            continue

        # exclude run if less than half of the frames remained after censoring
        run_length = 383
        for run in range(int(total_frames/run_length)):
            remaining_frames = 383 - removed_frames[range(run*383, (run+1)*383)].sum()
            if remaining_frames/383 < 0.5:
                print(
                    f'{subject} has {remaining_frames} frames left for run {run} after censoring',
                    file=log)
                continue
            else:
                sublist.append(f'{subject}_run-{run+1}')

sublist = [f'sub-NDAR{subject.lstrip("NDAR_")}' for subject in sublist]
pd.Series(sublist).to_csv(args.output, header=False, index=False)

if args.log is not None:
    log.close()
