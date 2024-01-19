#!/usr/bin/env bash

work_dir=$(pwd)/work
out_dir=$(pwd)/hcpdiff_output
simg=$(pwd)/mpp/replication/singularity_files/mdiffusion.simg
get_cmd="datalad get -s inm7-storage"

main(){
    mkdir -p $work_dir
    for subject in `cat $sublist`; do
        sub_dir=${dataset_dir}/${subject}
        $get_cmd -n -d $dataset_dir $sub_dir

        d_dir=$sub_dir/unprocessed/Diffusion
        $get_cmd -n -d $sub_dir $sub_dir/unprocessed
        for ndir in 98 99; do
            for phase in AP PA; do
                for file_type in .nii.gz .bval .bvec; do
                    key=dir${ndir}_${phase}${file_type}
                    $get_cmd -d $sub_dir/unprocessed $d_dir/${subject}_dMRI_${key}
                done
            done
        done

        anat_dir=$sub_dir/T1w
        $get_cmd -n -d $sub_dir $anat_dir
        $get_cmd -d $anat_dir $anat_dir/T1w_acpc_dc.nii.gz
        $get_cmd -d $anat_dir $anat_dir/T1w_acpc_dc_restore.nii.gz
        $get_cmd -d $anat_dir $anat_dir/T1w_acpc_dc_restore_brain.nii.gz
        $get_cmd -d $anat_dir $anat_dir/BiasField_acpc_dc.nii.gz
        $get_cmd -d $anat_dir $anat_dir/brainmask_fs.nii.gz

        fs_dir=$sub_dir/T1w/$subject
        $get_cmd -d $anat_dir $fs_dir/surf/lh.white.deformed
        $get_cmd -d $anat_dir $fs_dir/surf/rh.white.deformed
        $get_cmd -d $anat_dir $fs_dir/mri/transforms/eye.dat
        $get_cmd -d $anat_dir $fs_dir/mri/orig.mgz
        $get_cmd -d $anat_dir $fs_dir/surf/lh.thickness
        $get_cmd -d $anat_dir $fs_dir/surf/rh.thickness

        command="hcpdiffpy $dataset_dir/$subject $subject 0.69 --ndirs 98 99 --phases AP PA \
                --work_dir $work_dir --output_dir $out_dir/$subject \
                --fsl_simg $simg --fs_simg $simg --wb_simg $simg --condordag"
        echo $command
        eval $command
    done
}

usage(){ echo "
Usage: $0 -d dataset_dir -l sublist

REQUIRED ARGUMENT:
    -d dataset_dir  absolute path to dataset directory
    -l sublist      absolute path to subject list (.csv)
" 1>&2; exit 1;}

while getopts "d:l:h" opt; do
    case $opt in
        d) dataset_dir=${OPTARG} ;;
        l) sublist=${OPTARG} ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [ -z $dataset_dir ]; then echo "Dataset directory not defined"; 1>&2; exit 1; fi
if [ -z $sublist ]; then echo "Subject list not defined"; 1>&2; exit 1; fi

main
