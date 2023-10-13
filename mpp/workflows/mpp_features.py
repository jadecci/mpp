from pathlib import Path
import argparse
import logging
from typing import Union

import pandas as pd
import nipype.pipeline as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
from nipype import config

from mpp.interfaces.data import InitData, SaveFeatures, InitDiffusionData, SaveDFeatures
from mpp.interfaces.features import (
    RSFC, NetworkStats, TFC, MyelinEstimate, Morphometry, SCWF, FAMD)
from mpp.interfaces.preproc import HCPMinProc, CSD, TCK
from mpp.utilities.utilities import SimgCmd
from mpp.utilities.preproc import d_files_intermediate

logging.getLogger('datalad').setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Multimodal psychometric prediction feature extraction',
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
    parser.add_argument('dataset', type=str, help='Dataset (HCP-YA, HCP-A, HCP-D, ABCD)')
    parser.add_argument('sublist', type=str, help='Absolute path to the subject list (.csv).')
    parser.add_argument(
        '--diffusion', dest='diffusion', action='store_true',
        help='Process diffusion data and extract diffusion features')
    parser.add_argument(
        '--intermediate_dir', type=Path, dest='int_dir', default=None,
        help='Directory containing intermediate files from previous runs')
    parser.add_argument(
        '--workdir', type=Path, dest='work_dir', default=Path.cwd(), help='Work directory')
    parser.add_argument(
        '--output_dir', type=Path, dest='output_dir', default=Path.cwd(), help='Output directory')
    parser.add_argument(
        '--simg', type=Path, dest='simg', default=None,
        help='singularity image to use for command line functions from FSL / FreeSurfer / '
             'Connectome Workbench / MRTrix3.')
    parser.add_argument(
        '--overwrite', dest='overwrite', action="store_true", help='Overwrite existing results')
    parser.add_argument(
        '--condordag', dest='condordag', action='store_true',
        help='Submit graph workflow to HTCondor')
    parser.add_argument(
        '--wrapper', type=str, dest='wrapper', default='', help='Wrapper script for HTCondor')
    parser.add_argument(
        '--debug', dest='debug', action="store_true", help='Use debug configuration')
    args = parser.parse_args()

    simg_cmd = SimgCmd(
        simg=args.simg, work_dir=args.work_dir, out_dir=args.output_dir, int_dir=args.int_dir)

    sublist = pd.read_csv(args.sublist, header=None).squeeze('columns')
    for subject in sublist:
        output_file = Path(args.output_dir, f'{args.dataset}_{subject}.h5')
        if not output_file.is_file() or args.overwrite:
            if args.diffusion:
                subject_wf = init_d_wf(
                    args.dataset, str(subject), args.work_dir, args.output_dir, simg_cmd,
                    args.overwrite, args.int_dir)
            else:
                subject_wf = init_subject_wf(
                    args.dataset, str(subject), args.work_dir, args.output_dir, simg_cmd,
                    args.overwrite)

            subject_wf.config['execution']['try_hard_link_datasink'] = 'false'
            subject_wf.config['execution']['crashfile_format'] = 'txt'
            subject_wf.config['execution']['stop_on_first_crash'] = 'true'
            subject_wf.config['monitoring']['enabled'] = 'true'
            if args.debug:
                config.enable_debug_mode()

            subject_wf.write_graph()
            if args.condordag:
                subject_wf.run(
                    plugin='CondorDAGMan',
                    plugin_args={
                        'dagman_args': f'-outfile_dir {args.work_dir}', 'wrapper_cmd': args.wrapper,
                        'dagman_args': '-import_env',
                        'override_specs': 'request_memory = 5 GB\nrequest_cpus = 1'})
            else:
                subject_wf.run()


def init_subject_wf(
        dataset: str, subject: str, work_dir: Path, output_dir: Path, simg_cmd: SimgCmd,
        overwrite: bool) -> pe.Workflow:
    subject_wf = pe.Workflow(f'subject_{subject}_wf', base_dir=work_dir)
    work_curr = Path(work_dir, f'subject_{subject}_wf')
    init_data = pe.Node(
        InitData(
            dataset=dataset, work_dir=work_curr, subject=subject, output_dir=output_dir,
            simg_cmd=simg_cmd),
        name='init_data')
    save_features = pe.Node(
        SaveFeatures(output_dir=output_dir, dataset=dataset, subject=subject, overwrite=overwrite),
        name='save_features')

    rs_wf = init_rs_wf(dataset, subject)
    anat_wf = init_anat_wf(dataset, subject, work_curr, simg_cmd)

    subject_wf.connect([
        (init_data, rs_wf, [
            ('rs_runs', 'inputnode.rs_runs'), ('rs_files', 'inputnode.rs_files'),
            ('hcpd_b_runs', 'inputnode.hcpd_b_runs')]),
        (init_data, anat_wf, [
            ('anat_dir', 'inputnode.anat_dir'), ('anat_files', 'inputnode.anat_files')]),
        (init_data, save_features, [('dataset_dir', 'dataset_dir')]),
        (rs_wf, save_features, [
            ('outputnode.rsfc', 'rsfc'), ('outputnode.dfc', 'dfc'), ('outputnode.efc', 'efc'),
            ('outputnode.rs_stats', 'rs_stats')]),
        (anat_wf, save_features, [('outputnode.myelin', 'myelin'), ('outputnode.morph', 'morph')])])

    # task features are only extracted for HCP subjects
    if 'HCP' in dataset:
        t_wf = init_t_wf(dataset, subject)
        subject_wf.connect([
            (init_data, t_wf, [('t_runs', 'inputnode.t_runs'), ('t_files', 'inputnode.t_files')]),
            (t_wf, save_features, [('outputnode.tfc', 'tfc')])])

    return subject_wf


def init_rs_wf(dataset: str, subject: str) -> pe.Workflow:
    rs_wf = pe.Workflow(f'subject_{subject}_rs_wf')
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['rs_runs', 'rs_files', 'hcpd_b_runs']), name='inputnode')
    rsfc = pe.Node(RSFC(dataset=dataset), name='rsfc')
    network_stats = pe.Node(NetworkStats(), name='network_stats')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['rsfc', 'dfc', 'efc', 'rs_stats']), name='outputnode')

    rs_wf.connect([
        (inputnode, rsfc, [
            ('rs_runs', 'rs_runs'), ('rs_files', 'rs_files'), ('hcpd_b_runs', 'hcpd_b_runs')]),
        (rsfc, network_stats, [('rsfc', 'conn')]),
        (rsfc, outputnode, [('rsfc', 'rsfc'), ('dfc', 'dfc'), ('efc', 'efc')]),
        (network_stats, outputnode, [('stats', 'rs_stats')])])

    return rs_wf


def init_t_wf(dataset: str, subject: str) -> pe.Workflow:
    t_wf = pe.Workflow(f'subject_{subject}_t_wf')
    inputnode = pe.Node(niu.IdentityInterface(fields=['t_runs', 't_files']), name='inputnode')
    tfc = pe.Node(TFC(dataset=dataset), name='tfc')
    outputnode = pe.Node(niu.IdentityInterface(fields=['tfc']), name='outputnode')

    t_wf.connect([
        (inputnode, tfc, [('t_runs', 't_runs'), ('t_files', 't_files')]),
        (tfc, outputnode, [('tfc', 'tfc')])])

    return t_wf


def init_anat_wf(dataset: str, subject: str, work_dir: Path, simg_cmd: SimgCmd) -> pe.Workflow:
    anat_wf = pe.Workflow(f'subject_{subject}_anat_wf')
    work_curr = Path(work_dir, f'subject_{subject}_anat_wf')
    inputnode = pe.Node(niu.IdentityInterface(fields=['anat_dir', 'anat_files']), name='inputnode')
    myelin = pe.Node(MyelinEstimate(), name='myelin')
    morphometry = pe.Node(
        Morphometry(dataset=dataset, subject=subject, work_dir=work_curr, simg_cmd=simg_cmd),
        name='morphometry')
    outputnode = pe.Node(niu.IdentityInterface(fields=['myelin', 'morph']), name='outputnode')

    anat_wf.connect([
        (inputnode, morphometry, [('anat_dir', 'anat_dir'), ('anat_files', 'anat_files')]),
        (inputnode, myelin, [('anat_files', 'anat_files')]),
        (myelin, outputnode, [('myelin', 'myelin')]),
        (morphometry, outputnode, [('morph', 'morph')])])

    return anat_wf


def init_d_wf(
        dataset: str, subject: str, work_dir: Path, output_dir: Path, simg_cmd: SimgCmd,
        overwrite: bool, int_dir: Union[Path, None]) -> pe.Workflow:
    d_wf = pe.Workflow(f'subject_{subject}_diffusion_wf', base_dir=work_dir)
    work_curr = Path(work_dir, f'subject_{subject}_diffusion_wf')
    init_data = pe.Node(
        InitDiffusionData(
            dataset=dataset, work_dir=work_curr, subject=subject, output_dir=output_dir),
        name='init_data')
    if int_dir is not None:
        init_data.inputs.int_dir = Path(int_dir, subject)

    tmp_dir = Path(work_curr, 'intermediate_tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dtifit = pe.Node(fsl.DTIFit(command=simg_cmd.run_cmd('dtifit')), name='dtifit')
    csd = pe.Node(CSD(dataset=dataset, work_dir=tmp_dir, simg_cmd=simg_cmd), name='csd')
    tck = pe.Node(TCK(dataset=dataset, work_dir=tmp_dir, simg_cmd=simg_cmd), name='tck')

    sc = SCWF(subject=subject, dataset=dataset, work_dir=work_curr, simg_cmd=simg_cmd).run()
    sc_wf = sc.outputs.sc_wf
    sc_length_stats = pe.Node(NetworkStats(), name='sc_length_stats')
    sc_count_stats = pe.Node(NetworkStats(), name='sc_count_stats')
    # fa_md = pe.Node(FAMD(), name='fa_md')

    save_features = pe.Node(
        SaveDFeatures(output_dir=output_dir, dataset=dataset, subject=subject, overwrite=overwrite),
        name='save_features')

    if (dataset == 'HCP-A' or dataset == 'HCP-D') and int_dir is None:
        hcp_proc = HCPMinProc(
            dataset=dataset, work_dir=work_curr, subject=subject, simg_cmd=simg_cmd).run()
        hcp_proc_wf = hcp_proc.outputs.hcp_proc_wf
        d_wf.connect([
            (init_data, hcp_proc_wf, [
                ('d_files', 'inputnode.d_files'), ('t1_files', 'inputnode.t1_files'),
                ('fs_files', 'inputnode.fs_files'), ('fs_dir', 'inputnode.fs_dir')]),
            (hcp_proc_wf, dtifit, [
                ('outputnode.data', 'dwi'), ('outputnode.bval', 'bvals'),
                ('outputnode.bvec', 'bvecs'), ('outputnode.mask', 'mask')]),
            (hcp_proc_wf, csd, [
                ('outputnode.data', 'data'), ('outputnode.bval', 'bval'),
                ('outputnode.bvec', 'bvec'), ('outputnode.mask', 'mask')]),
            (hcp_proc_wf, tck, [('outputnode.bval', 'bval'), ('outputnode.bvec', 'bvec')])])

    elif dataset == 'HCP-YA' or int_dir is not None:
        split_files = pe.Node(
            niu.Function(
                function=d_files_intermediate, output_names=['data', 'bval', 'bvec', 'mask']),
            name='split_files')
        d_wf.connect([
            (init_data, split_files, [('d_files', 'd_files')]),
            (split_files, dtifit, [
                ('data', 'dwi'), ('bval', 'bvals'), ('bvec', 'bvecs'), ('mask', 'mask')]),
            (split_files, csd, [
                ('data', 'data'), ('bval', 'bval'), ('bvec', 'bvec'), ('mask', 'mask')]),
            (split_files, tck, [('bval', 'bval'), ('bvec', 'bvec')])])

    d_wf.connect([
        (init_data, tck, [('fs_dir', 'fs_dir')]),
        (csd, tck, [('fod_wm_file', 'fod_wm_file')]),
        (init_data, sc_wf, [
            ('t1_files', 'inputnode.t1_files'), ('fs_files', 'inputnode.fs_files'),
            ('fs_dir', 'inputnode.fs_dir')]),
        (tck, sc_wf, [('tck_file', 'inputnode.tck_file')]),
        (sc_wf, sc_count_stats, [('outputnode.count_files', 'conn_files')]),
        (sc_wf, sc_length_stats, [('outputnode.length_files', 'conn_files')]),
        # (dtifit, fa_md, [('FA', 'fa_file'), ('MD', 'md_file')]),
        # (sc_wf, fa_md, [('outputnode.atlas_files', 'atlas_files')]),
        (init_data, save_features, [('dataset_dir', 'dataset_dir')]),
        (sc_wf, save_features, [
            ('outputnode.count_files', 'count_files'),
            ('outputnode.length_files', 'length_files')]),
        (sc_count_stats, save_features, [('stats', 'count_stats')]),
        (sc_length_stats, save_features, [('stats', 'length_stats')])])
        # (fa_md, save_features, [('fa', 'fa'), ('md', 'md')])])

    return d_wf


if __name__ == '__main__':
    main()
