## 1. Preparation

1. Download the phenotype files into the `phenotype` folder, following the directory structure.
2. Generate the subject lists for feature extraction
```python
# HCP-YA, HCP-A and HCP-D
python3 sublist/create_allRun_sublists.py
# ABCD
datalad clone git@jugit.fz-juelich.de:inm7/datasets/datasets_repo.git datasets_repo
datalad get -d datasets_repo -n datasets_repo/original/abcd
python3 sublist/create_sublist_ABCD.py datasets_repo/original/abcd sublist/ABCD.csv \
        --source inm7-storage --log sublist/ABCD.log
datalad remove -d datasets_repo --reckless kill
``` 

## 2. Features Extraction

1. Create a singularity container for diffusion processing (done in macOS Ventura)

```bash
mkdir vm-singularity-ce && cp singularity_files/* vm-singularity-ce/ && cd vm-singularity-ce
vagrant init sylabs/singularity-ce-3.9-ubuntu-bionic64 
# if plugin not installed yet: vagrant plugin install vagrant-disksize
# then add to Vagrantfile: config.disksize.size = '50GB'
vagrant up && vagrant ssh
sudo singularity build mdiffusion /vagrant/mdiffusion.def
```