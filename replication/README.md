## 1. Preparation

### 1.1. Install dataset
Clone and install from GIN repository to get all the replication files

```bash
python3 -m venv ~/.venvs/mpp-features
source ~/.venvs/mpp-features/bin/activate
datalad clone git@gin.g-node.org:/jadecci/MPP.git ${project_dir}/mpp
cd ${project_dir}/mpp && datalad get -r . && cd ${project_dir}
python3 install ${project_dir}/mpp
```

### 1.2. Download phenotype files
Put the phenotype files into the `phenotype` folder, following the directory structure.

```console
${project_dir}/phenotype
├── HCP-A
├── HCP-D
├── restricted_hcpya.csv
└── unrestricted_hcpya.csv
```

## 2. Feature Extraction

### 2.1. Generate subject lists

```bash
python3 ${project_dir}/mpp/replication/sublist/create_allRun_sublists.py \
    ${project_dir}/phenotype ${project_dir}/sublist \
    ${project_dir}/mpp/replication/sublist/HCP-A_exclude.csv \
    ${project_dir}/mpp/replication/sublist/HCP-D_exclude.csv \
    ${project_dir}/mpp/replication/sublist/HCP-YA_exclude.csv
```

### 2.2. Preprocess diffusion data (HCP-A & HCP-D)

```bash
# For INM7 member on juseless
datalad clone git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git
datalad get -n -d datasets_repo datasets_repo/original

for subject in `cat sublist/HCP-A_allRun.csv`; do
    ${project_dir}/mpp/replication/submit_hcpdiff.sh \
        -d ${project_dir}/datasets_repo/original/hcp/hcp_aging \
        -l ${project_dir}/sublist/HCP-A_allRun.csv
done

for subject in `cat sublist/HCP-D_allRun.csv`; do
    ${project_dir}/mpp/replication/submit_hcpdiff.sh \
        -d ${project_dir}/datasets_repo/original/hcp/hcp_development \
        -l ${project_dir}/sublist/HCP-D_allRun.csv
done

# Common usage
simg=${project_dir}/mpp/replication/singularity_files/mdiffusion.simg

for subject in `cat ${project_dir}/sublist/HCP-A_allRun.csv`; do
    hcpdiffpy ${hcpa_data}/${subject} ${subject} 0.69 --ndirs 98 99 --phases AP PA \
        --work_dir ${project_dir}/work --output_dir ${project_dir}/hcpdiff_output \
        --fsl_simg ${simg} --fs_simg ${simg} --wb_simg ${simg}
done

for subject in `cat ${project_dir}/sublist/HCP-D_allRun.csv`; do
    hcpdiffpy ${hcpd_data}/${subject} ${subject} 0.69 --ndirs 98 99 --phases AP PA \
        --work_dir ${project_dir}/work --output_dir ${project_dir}/hcpdiff_output \
        --fsl_simg ${simg} --fs_simg ${simg} --wb_simg ${simg}
done
```

### 2.3. Extract DTI features

```bash
simg=${project_dir}/mpp/replication/singularity_files/mdiffusion.simg

mfe_dti HCP-YA ${project_dir}/sublist/HCP-YA_allRun.csv --simg ${simg} \
    --work_dir ${project_dir}/work --output_dir ${project_dir}/mfe_output

```

### 2.4. Extract (non-DTI) multimodal features

```bash
simg=${project_dir}/mpp/replication/singularity_files/mdiffusion.simg

for subject in cat `${project_dir}/sublist/HCP-YA_allRun.csv`; do
    mfe HCP-YA ${subject} --modality rfMRI tfMRI sMRI dMRI \
        --pheno_dir ${project_dir}/mpp/replication/phenotype \
        --work_dir ${project_dir}/work --output_dir ${project_dir}/mfe_output/${dataset}  \
        --simg ${simg}
done

for dataset in HCP-A HCP-D; do
    for subject in cat `${project_dir}/sublist/${dataset}_allRun.csv`; do
        mfe ${dataset} ${subject} --modality rfMRI tfMRI sMRI dMRI \
            --pheno_dir ${project_dir}/mpp/replication/phenotype/${dataset} \
            --work_dir ${project_dir}/work --output_dir ${project_dir}/mfe_output/${dataset} \
            --simg ${simg}
    done
done
```

## 3. Multimodal Prediction

### 3.1. Run predictions

```bash
mpp --datasets HCP-YA \
    --targets totalcogcomp crycogcomp fluidcogcomp cardsort flanker reading picvocab procspeed \
        listsort anger fear sadness posaffect emotsupp friendship loneliness neoffi_n neoffi_e \
        neoffi_o neoffi_a neoffi_c \
    --features_dir ${project_dir}/mfe_output \
    --sublists ${project_dir}/sublist/HCP-YA_allRun.csv \
    --level 3 \
    --hcpya_res ${project_dir}/mpp/replication/phenotype/restricted_hcpya.csv \
    --work_dir ${project_dir}/work \
    --output_dir ${project_dir}/mpp_output/${dataset}

for dataset in HCP-A HCP-D; do
    mpp --datasets $dataset \
        --targets totalcogcomp crycogcomp fluidcogcomp cardsort flanker reading picvocab procspeed \
            listsort anger fear sadness posaffect emotsupp friendship loneliness neoffi_n neoffi_e \
            neoffi_o neoffi_a neoffi_c \
        --features_dir ${project_dir}/mfe_output \
        --sublists ${project_dir}/sublist/${dataset}_allRun.csv \
        --level 3 \
        --work_dir ${project_dir}/work \
        --output_dir ${project_dir}/mpp_output/${dataset}
done
```

### 3.2. Plot results

Collect prediction results into tables for plotting:

```bash
python3 ${project_dir}/mpp/replication/figures/collect_results.py \
    --datasets HCP-A HCP-YA HCP-D \
    --pred_dir ${project_dir}/mpp_output \
    --out_dir ${project_dir}/figures
```

Plot all figures:

```bash
python3 ${project_dir}/mpp/replication/figures/plot_figures.py \
    --res_dir ${project_dir}/figures \
    --out_dir ${project_dir}/figures
```

## Additional information

### 1. Creation of singularity containers

To create a singularity container for diffusion processing (done in macOS Ventura):

```bash
cd singularity-files
vagrant init sylabs/singularity-ce-3.9-ubuntu-bionic64
# if plugin not installed yet
vagrant plugin install vagrant-disksize 
# then add to Vagrantfile: 
# config.disksize.size = '100GB'
# config.vm.provider 'virtualbox' do |vb|
#     vb.memory = '2048'
# end
vagrant up && vagrant ssh
# add the available disk space
sudo cfdisk /dev/sda #[Resize] /dev/sda1 [Write]
sudo pvresize /dev/sda1
sudo lvextend -r -l +100%FREE /dev/mapper/vagrant--vg-root
# build singularity image
sudo singularity build mdiffusion.simg /vagrant/mdiffusion.def
mv mdiffusion.simg /vagrant/mdiffusion.simg
```
