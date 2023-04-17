import pandas as pd
from pathlib import Path
import argparse
import logging

import nipype.pipeline as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl

from mpp.interfaces.data import InitData, SaveFeatures, DropSubData, InitDiffusionData, SaveDFeatures
from mpp.interfaces.features import RSFC, NetworkStats, TFC, MyelinEstimate, Morphometry, SCWF
from mpp.interfaces.preproc import HCPMinProc, CSD, TCK

logging.getLogger('datalad').setLevel(logging.WARNING)

def main():
    parser = argparse.ArgumentParser(description='Multimodal psychometric prediction feature extraction',
                formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
    parser.add_argument('dataset', type=str, help='Dataset (HCP-YA, HCP-A, HCP-D, ABCD)')
    parser.add_argument('sublist', type=str, help='Absolute path to the subject list (.csv).')
    parser.add_argument('--diffusion', dest='diffusion', action='store_true', 
                        help='Process diffusion data and extract diffusion features')
    parser.add_argument('--workdir', type=Path, dest='work_dir', default=Path.cwd(), help='Work directory')
    parser.add_argument('--output_dir', type=Path, dest='output_dir', default=Path.cwd(), help='output directory')
    parser.add_argument('--overwrite', dest='overwrite', action="store_true", help='overwrite existing results')
    parser.add_argument('--condordag', dest='condordag', action='store_true', help='submit graph workflow to HTCondor')
    parser.add_argument('--wrapper', type=str, dest='wrapper', default='', help='wrapper script for HTCondor')
    args = parser.parse_args()

    sublist = pd.read_csv(args.sublist, header=None, squeeze=True)
    for subject in sublist:
        output_file = Path(args.output_dir, f'{args.dataset}_{subject}.h5')
        if not output_file.is_file() or args.overwrite:
            if args.diffusion:
                subject_wf = init_d_wf(args.dataset, subject, args.work_dir, args.output_dir, args.overwrite)
            else:
                subject_wf = init_subject_wf(args.dataset, subject, args.work_dir, args.output_dir, args.overwrite)
            subject_wf.config['execution']['try_hard_link_datasink'] = 'false'
            subject_wf.config['execution']['crashfile_format'] = 'txt'
            subject_wf.config['execution']['stop_on_first_crash'] = 'true'
            subject_wf.config['monitoring']['enabled'] = 'true'

            subject_wf.write_graph()
            if args.condordag:
                subject_wf.run(plugin='CondorDAGMan', plugin_args={'dagman_args': f'-outfile_dir {args.work_dir}',
                                                                'wrapper_cmd': args.wrapper})
            else:
                subject_wf.run()
                                                  
def init_subject_wf(dataset, subject, work_dir, output_dir, overwrite):
    subject_wf = pe.Workflow(f'subject_{subject}_wf', base_dir=work_dir)

    init_data = pe.Node(InitData(dataset=dataset, work_dir=work_dir, subject=subject, output_dir=output_dir),
                                 name='init_data')
    save_features = pe.Node(SaveFeatures(output_dir=output_dir, dataset=dataset, subject=subject, overwrite=overwrite),
                            name='save_features')

    rs_wf = init_rs_wf(dataset, subject)
    anat_wf = init_anat_wf(subject)

    subject_wf.connect([(init_data, rs_wf, [('rs_runs', 'inputnode.rs_runs'),
                                            ('rs_files', 'inputnode.rs_files'),
                                            ('hcpd_b_runs', 'inputnode.hcpd_b_runs')]),
                         (init_data, anat_wf, [('anat_dir', 'inputnode.anat_dir'),
                                               ('anat_files', 'inputnode.anat_files')]),
                         (init_data, save_features, [('dataset_dir', 'dataset_dir')]),
                         (rs_wf, save_features, [('outputnode.rsfc', 'rsfc'),
                                                 ('outputnode.dfc', 'dfc'),
                                                 ('outputnode.rs_stats', 'rs_stats')]),
                         (anat_wf, save_features, [('outputnode.myelin', 'myelin'),
                                                   ('outputnode.morph', 'morph')])])

    # task features are only extracted for HCP subjects
    if 'HCP' in dataset:
        t_wf = init_t_wf(dataset, subject)
        subject_wf.connect([(init_data, t_wf, [('t_runs', 'inputnode.t_runs'),
                                               ('t_files', 'inputnode.t_files')]),
                            (t_wf, save_features, [('outputnode.tfc', 'tfc')])])

    return subject_wf 

def init_rs_wf(dataset, subject):
    rs_wf = pe.Workflow(f'subject_{subject}_rs_wf')

    inputnode = pe.Node(niu.IdentityInterface(fields=['rs_runs', 'rs_files', 'hcpd_b_runs']), 
                                              name='inputnode')
    rsfc = pe.Node(RSFC(dataset=dataset), name='rsfc')
    network_stats = pe.Node(NetworkStats(), name='network_stats')
    outputnode = pe.Node(niu.IdentityInterface(fields=['rsfc', 'dfc', 'rs_stats']), name='outputnode')

    rs_wf.connect([(inputnode, rsfc, [('rs_runs', 'rs_runs'),
                                      ('rs_files', 'rs_files'),
                                      ('hcpd_b_runs', 'hcpd_b_runs')]),
                    (rsfc, network_stats, [('rsfc', 'rsfc')]),
                    (rsfc, outputnode, [('rsfc', 'rsfc'),
                                        ('dfc', 'dfc')]),
                    (network_stats, outputnode, [('rs_stats', 'rs_stats')])])

    return rs_wf

def init_t_wf(dataset, subject):
    t_wf = pe.Workflow(f'subject_{subject}_t_wf')

    inputnode = pe.Node(niu.IdentityInterface(fields=['t_runs', 't_files']), name='inputnode')
    tfc = pe.Node(TFC(dataset=dataset), name='tfc')
    outputnode = pe.Node(niu.IdentityInterface(fields=['tfc']), name='outputnode')

    t_wf.connect([(inputnode, tfc, [('t_runs', 't_runs'),
                                    ('t_files', 't_files')]),
                  (tfc, outputnode, [('tfc', 'tfc')])])

    return t_wf

def init_anat_wf(subject):
    anat_wf = pe.Workflow(f'subject_{subject}_anat_wf')

    inputnode = pe.Node(niu.IdentityInterface(fields=['anat_dir', 'anat_files']), name='inputnode')
    myelin = pe.Node(MyelinEstimate(), name='myelin')
    morphometry = pe.Node(Morphometry(subject=subject), name='morphometry')
    outputnode = pe.Node(niu.IdentityInterface(fields=['myelin', 'morph']), name='outputnode')

    anat_wf.connect([(inputnode, morphometry, [('anat_dir', 'anat_dir'),
                                               ('anat_files', 'anat_files')]),
                     (inputnode, myelin, [('anat_files', 'anat_files')]),
                     (myelin, outputnode, [('myelin', 'myelin')]),
                     (morphometry, outputnode, [('morph', 'morph')])])

    return anat_wf

def init_d_wf(dataset, subject, work_dir, output_dir, overwrite):
    d_wf = pe.Workflow(f'subject_{subject}_diffusion_wf', base_dir=work_dir)

    init_data = pe.Node(InitDiffusionData(dataset=dataset, work_dir=work_dir, subject=subject, output_dir=output_dir),
                                 name='init_data')
    
    hcp_proc = HCPMinProc(dataset=dataset, work_dir=work_dir, subject=subject).run()
    hcp_proc_wf = hcp_proc.outputs.hcp_proc_wf

    tmp_dir = Path(work_dir, 'intermediate_tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dtifit = pe.Node(fsl.DTIFit(), name='dtifit')
    csd = pe.Node(CSD(dataset=dataset, work_dir=tmp_dir), name='csd')
    tck = pe.Node(TCK(work_dir=tmp_dir), name='tck')

    sc = SCWF(subject=subject, work_dir=work_dir).run()
    sc_wf = sc.outputs.sc_wf

    save_features = pe.Node(SaveDFeatures(output_dir=output_dir, dataset=dataset, subject=subject, overwrite=overwrite),
                            name='save_features')

    d_wf.connect([(init_data, hcp_proc_wf, [('dataset_dir', 'inputnode.dataset_dir'),
                                            ('d_files', 'inputnode.d_files'),
                                            ('fs_files', 'inputnode.fs_files')]),
                  (hcp_proc_wf, dtifit, [('outputnode.data', 'dwi'),
                                         ('outputnode.bval', 'bvals'),
                                         ('outputnode.bvec', 'bvecs'),
                                         ('outputnode.mask', 'mask')]),
                  (hcp_proc_wf, csd, [('outputnode.data', 'data'),
                                      ('outputnode.bval', 'bval'),
                                      ('outputnode.bvec', 'bvec'),
                                      ('outputnode.mask', 'mask')]),
                  (init_data, tck, [('fs_dir', 'fs_dir')]),
                  (hcp_proc_wf, tck, [('outputnode.bval', 'bval'),
                                      ('outputnode.bvec', 'bvec')]),
                  (csd, tck, [('fod_wm_file', 'fod_wm_file')]),
                  (init_data, sc_wf, [('t1_file', 'inputnode.t1_file'),
                                      ('fs_files', 'inputnode.fs_files'),
                                      ('fs_dir', 'inputnode.fs_dir')]),
                  (tck, sc_wf, [('tck_file', 'inputnode.tck_file')]),
                  (init_data, save_features, [('dataset_dir', 'dataset_dir')]),
                  (sc_wf, save_features, [('outputnode.count_files', 'count_files'),
                                          ('outputnode.length_files', 'length_files')])])
        
    return d_wf

if __name__ == '__main__':
    main()