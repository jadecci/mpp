## 1. Preparation & Featrues Extraction
Clone and install from GIN repository to get all the replication data files and the container
```bash
python3 -m venv ~/.venvs/mpp-features
source ~/.venvs/mpp-features/bin/activate
datalad clone git@gin.g-node.org:/jadecci/MPP.git ${project_dir}/mpp
cd ${project_dir}/mpp && datalad get -r . && cd ${project_dir}
```

### 1.1. HCP-YA, HCP-A and HCP-D
1. Download the phenotype files into the `phenotype` folder, following the directory structure.
2. Generate the subject lists for feature extraction
```bash
python3 ${project_dir}/mpp/sublist/create_allRun_sublists.py
```
1. Preprocess diffusion data for HCP Aging and HCP Development
```bash
# For INM7 member on juseless
datalad clone git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git
datalad get -n -d datasets_repo datasets_repo/original
for subject in `cat sublist/HCP-A_allRun.csv`; do
    ./mpp/replication/submit_hcpdiff.sh -d ${project_dir}/datasets_repo/original/hcp/hcp_aging \
        -l ${project_dir}/sublist/HCP-A_allRun.csv
done
for subject in `cat sublist/HCP-D_allRun.csv`; do
    ./mpp/replication/submit_hcpdiff.sh -d ${project_dir}/datasets_repo/original/hcp/hcp_development \
        -l ${project_dir}/sublist/HCP-D_allRun.csv
done

# Common usage
simg=${project_dir}/mpp/replication/singularity_files/mdiffusion.simg
for subject in `cat sublist/HCP-A_allRun.csv`; do
    hcpdiffpy ${hcpa_data}/${subject} ${subject} 0.69 --ndirs 98 99 --phases AP PA \
        --work_dir $(pwd)/work --output_dir $(pwd)/hcpdiff_output \
        --fsl_simg ${simg} --fs_simg ${simg} --wb_simg ${simg}
done
for subject in `cat sublist/HCP-D_allRun.csv`; do
    hcpdiffpy ${hcpd_data}/${subject} ${subject} 0.69 --ndirs 98 99 --phases AP PA \
        --work_dir $(pwd)/work --output_dir $(pwd)/hcpdiff_output \
        --fsl_simg ${simg} --fs_simg ${simg} --wb_simg ${simg}
done
```
1. Extract multimodal features
```bash
for dataset in HCP-YA, HCP-A, HCP-D; do
    mfeatures ${dataset} sublist/${dataset}_allRun.csv --output_dir ${dataset}/features \
        --condordag --wrapper venv_wrapper.sh
done
```


### 1.2. ABCD
```bash
# ABCD
datalad clone git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git datasets_repo
datalad get -d datasets_repo -n datasets_repo/original/abcd
python3 sublist/create_sublist_ABCD.py datasets_repo/original/abcd sublist/ABCD.csv \
        --source inm7-storage --log sublist/ABCD.log
datalad remove -d datasets_repo --reckless kill
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
