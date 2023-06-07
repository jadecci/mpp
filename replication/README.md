## 1. Preparation & Featrues Extraction
```bash
python3 -m venv ~/.venvs/dev-multipred
source ~/.venvs/dev-multipred/bin/activate
python3 -m pip install git+https://github.com/jadecci/mpp.git@v3.0
```

### 1.1. HCP-YA, HCP-A and HCP-D
1. Download the phenotype files into the `phenotype` folder, following the directory structure.
2. Generate the subject lists for feature extraction
```bash
python3 sublist/create_allRun_sublists.py
```
3. Features extraction for task, rest and structural data
```bash
for dataset in HCP-YA, HCP-A, HCP-D; do
    mfeatures ${dataset} sublist/${dataset}_allRun.csv --output_dir features/${dataset} \
              --condordag --wrapper venv_wrapper.sh
done
```
4. Features extraction for diffusion data (step 3 needs to finish running first)
```bash
for dataset in HCP-YA, HCP-A, HCP-D; do
    mfeatures ${dataset} sublist/${dataset}_allRun.csv --output_dir features/${dataset} \
              --diffusion --condordag --wrapper venv_wrapper.sh
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
sudo cfdisk /dev/sda
sudo pvresize /dev/sda1
sudo lvextend -r -l +100%FREE /dev/mapper/vagrant--vg-root
# build singularity image
sudo singularity build mdiffusion.simg /vagrant/mdiffusion.def
mv mdiffusion.simg /vagrant/mdiffusion.simg
```