import nipype.pipeline as pe
from nipype.interfaces import utility as niu
from nipype import config
import pandas as pd
from os import getcwd, path
import argparse
import logging

from mpp.interfaces.data import InitData, InitSubData, InitRSData, InitAnatData, RSSave, AnatSave, DropSubData
from mpp.interfaces.processing import NuisanceReg, ParcellateTimeseries
from mpp.interfaces.features import RSFC, DFC, NetworkStats, MyelinEstimate, Morphometry

base_dir = path.join(path.dirname(path.realpath(__file__)), '..', '..')
logging.getLogger('datalad').setLevel(logging.WARNING)
config.update_config({'execution': {'stop_on_first_crash': 'true'}})

def main():
    parser = argparse.ArgumentParser(description='Multimodal psychometric prediction',
                formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
    parser.add_argument('dataset', type=str, help='Dataset (HCP-YA, HCP-A, HCP-D, ABCD)')
    parser.add_argument('--sublist', type=str, dest='sublist', default=None, 
                        help='Absolute path to the subject list (.csv). Default uses all subjects.')
    parser.add_argument('--workdir', type=str, dest='work_dir', default=getcwd(), help='Work directory')
    parser.add_argument('--overwrite', dest='overwrite', action="store_true", help='overwrite existing results')
    parser.add_argument('--wrapper', type=str, dest='wrapper', default='', help='wrapper script')
    args = parser.parse_args()

    dataset_sublist = {'HCP-YA': path.join(base_dir, ''),
                       'HCP-A': path.join(base_dir, ''),
                       'HCP-D': path.join(base_dir, ''),
                       'ABCD': path.join(base_dir, '')}

    ## Overall workflow
    mmf_wf = pe.Workflow('mpp_wf', base_dir=args.work_dir)
    init_data = pe.Node(InitData(dataset=args.dataset, work_dir=args.work_dir), name='init_data')

    ## Individual subject's workflow
    sublist = pd.read_csv(args.sublist, header=None, squeeze=True)
    for subject in sublist:
        subject_wf = init_subject_wf(args.dataset, subject, args.work_dir, args.overwrite)
        mmf_wf.connect(init_data, 'dataset_dir', subject_wf, 'inputnode.dataset_dir')

    mmf_wf.write_graph()
    mmf_wf.run(plugin='CondorDAGMan', plugin_args={'dagman_args': f'-outfile_dir {args.work_dir}',
                                                   'wrapper_cmd': args.wrapper})

def init_subject_wf(dataset, subject, work_dir, overwrite):
    wf_name = 'subject_%s_wf' % subject
    wf_dir = path.join(work_dir, subject)
    subject_wf = pe.Workflow(wf_name, base_dir=wf_dir)

    inputnode = pe.Node(niu.IdentityInterface(fields=['dataset_dir']), name='inputnode')
    init_data = pe.Node(InitSubData(dataset=dataset, subject=subject), name='init_data')
    drop_data = pe.Node(DropSubData(), name='drop_data')

    rs_wf = init_rs_wf(dataset, subject, work_dir, overwrite)
    anat_wf = init_anat_wf(dataset, subject, work_dir, overwrite)
    subject_wf.connect([(inputnode, init_data, [('dataset_dir', 'dataset_dir')]),
                         (init_data, rs_wf, [('rs_dir', 'inputnode.rs_dir'),
                                             ('rs_files', 'inputnode.rs_files'),
                                             ('rs_skip', 'inputnode.rs_skip')]),
                         (init_data, anat_wf, [('rs_dir', 'inputnode.rs_dir'),
                                               ('anat_dir', 'inputnode.anat_dir'),
                                               ('anat_files', 'inputnode.anat_files'),
                                               ('t1_skip', 'inputnode.t1_skip'),
                                               ('myelin_skip', 'inputnode.myelin_skip')]),
                         (init_data, drop_data, [('rs_dir', 'rs_dir'),
                                                 ('rs_files', 'rs_files'),
                                                 ('anat_dir', 'anat_dir'),
                                                 ('anat_files', 'anat_files')]),
                         (rs_wf, drop_data, [('outputnode.rs_done', 'rs_done')]),
                         (anat_wf, drop_data, [('outputnode.anat_done', 'anat_done')])])

    return subject_wf 

def init_rs_wf(dataset, subject, work_dir, overwrite):
    wf_name = 'subject_%s_rs_wf' % subject
    wf_dir = path.join(work_dir, subject, 'rest')
    rs_wf = pe.Workflow(wf_name, base_dir=wf_dir)

    inputnode = pe.Node(niu.IdentityInterface(fields=['rs_dir', 'rs_files', 'rs_skip']), name='inputnode')
    init_data = pe.Node(InitRSData(dataset=dataset), name='init_data')
    nuisance_reg = pe.Node(NuisanceReg(dataset=dataset), name='nuisance_reg')
    parcellate = pe.Node(ParcellateTimeseries(), name='parcellate')
    rsfc = pe.Node(RSFC(), name='rsfc')
    dfc = pe.Node(DFC(), name='dfc')
    network_stats = pe.Node(NetworkStats(), name='network_stats')
    save_features = pe.Node(RSSave(output_dir=work_dir, dataset=dataset, subject=subject, overwrite=overwrite), 
                            name='save_features')
    outputnode = pe.Node(niu.IdentityInterface(fields=['rs_done']), name='outputnode')

    rs_wf.connect([(inputnode, init_data, [('rs_dir', 'rs_dir'),
                                           ('rs_files', 'rs_files'),
                                           ('rs_skip', 'rs_skip')]),
                    (inputnode, nuisance_reg, [('rs_skip', 'rs_skip'),
                                               ('rs_dir', 'rs_dir')]),
                    (inputnode, parcellate, [('rs_skip', 'rs_skip')]),
                    (inputnode, rsfc, [('rs_skip', 'rs_skip')]),
                    (inputnode, network_stats, [('rs_skip', 'rs_skip')]),
                    (inputnode, save_features, [('rs_skip', 'rs_skip')]),
                    (init_data, nuisance_reg, [('rs_files', 'rs_files')]),
                    (nuisance_reg, parcellate, [('t_surf_resid', 't_surf_resid'),
                                                ('t_vol_resid', 't_vol_resid')]),
                    (parcellate, rsfc, [('tavg', 'tavg')]),
                    (parcellate, dfc, [('tavg', 'tavg')]),
                    (rsfc, network_stats, [('rsfc', 'rsfc')]),
                    (rsfc, save_features, [('rsfc', 'rsfc')]),
                    (dfc, save_features, [('dfc', 'dfc')]),
                    (network_stats, save_features, [('rs_stats', 'rs_stats')]),
                    (save_features, outputnode, [('rs_done', 'rs_done')])])

    return rs_wf

def init_anat_wf(dataset, subject, work_dir, overwrite):
    wf_name = 'subject_%s_anat_wf' % subject
    wf_dir = path.join(work_dir, subject, 'anat')
    anat_wf = pe.Workflow(wf_name, base_dir=wf_dir)

    inputnode = pe.Node(niu.IdentityInterface(fields=['rs_dir', 'anat_dir', 'anat_files', 't1_skip', 'myelin_skip']), 
                                              name='inputnode')
    init_data = pe.Node(InitAnatData(dataset=dataset), name='init_data')
    myelin = pe.Node(MyelinEstimate(), name='myelin')
    morphometry = pe.Node(Morphometry(subject=subject), name='morphometry')
    save_features = pe.Node(AnatSave(output_dir=work_dir, dataset=dataset, subject=subject, overwrite=overwrite),
                            name='save_features')
    outputnode = pe.Node(niu.IdentityInterface(fields=['anat_done']), name='outputnode')

    anat_wf.connect([(inputnode, init_data, [('rs_dir', 'rs_dir'),
                                             ('anat_dir', 'anat_dir'),
                                             ('anat_files', 'anat_files'),
                                             ('t1_skip', 't1_skip'),
                                             ('myelin_skip', 'myelin_skip')]),
                      (inputnode, myelin, [('myelin_skip', 'myelin_skip')]),
                      (inputnode, morphometry, [('t1_skip', 't1_skip'),
                                                ('anat_dir', 'anat_dir')]),
                      (inputnode, save_features, [('t1_skip', 't1_skip'),
                                                  ('myelin_skip', 'myelin_skip')]),
                      (init_data, myelin, [('anat_files', 'anat_files')]),
                      (init_data, morphometry, [('anat_files', 'anat_files')]),
                      (myelin, save_features, [('myelin', 'myelin')]),
                      (morphometry, save_features, [('morph', 'morph')]),
                      (save_features, outputnode, [('anat_done', 'anat_done')])])

    return anat_wf

if __name__ == '__main__':
    main()