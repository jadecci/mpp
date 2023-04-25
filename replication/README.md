## 1. Preparation

1. Download the phenotype files into the `phenotype` folder, following the directory structure.
2. Generate the subject lists for feature extraction
```bash
# HCP-YA, HCP-A and HCP-D
python3 sublist/create_allRun_sublists.py
# ABCD
datalad clone git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git datasets_repo
datalad get -d datasets_repo -n datasets_repo/original/abcd
python3 sublist/create_sublist_ABCD.py datasets_repo/original/abcd sublist/ABCD.csv \
        --source inm7-storage --log sublist/ABCD.log
datalad remove -d datasets_repo --reckless kill
``` 

## 2. Features extraction

## Additional information

### 1. Creation of singularity containers

To create a singularity container for diffusion processing (done in macOS Ventura):

```bash
cd singularity-files
#vagrant init sylabs/singularity-ce-3.9-ubuntu-bionic64 
vagrant init sylabs/singularity-2.6-ubuntu-bionic64
# if plugin not installed yet
vagrant plugin install vagrant-disksize 
# then add to Vagrantfile: 
# config.disksize.size = '100GB'
# config.vm.provider 'virtualbox' do |vb|
#     vb.memory = '2048'
# end
vagrant up && vagrant ssh
# add the available disk space
sudo cfdisk /dev/sda
sudo pvresize /dev/sda1
sudo lvextend -r -l +100%FREE /dev/mapper/vagrant--vg-root
# get debootstrap
sudo apt-get update
sudo apt-get install debootstrap
# build singularity image
sudo singularity build mdiffusion.simg /vagrant/mdiffusion.def
cp mdiffusion.simg /vagrant/mdiffusion.simg
```