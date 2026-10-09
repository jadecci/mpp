# Multimodal brain feature extraction pipeline

Two pipelines are implemented under this folder. First, `mfe` extract multimodal features from 
functional, structural and diffusion MRI for one subject. Second, `mfe_dti` extracts DTI-based 
features which require dataset-wide skeleton matching for one dataset.

## Multimodal feature extraction (`mfe`)

The workflow is defined in `main.py`

<img src="mfe_graph.png" height="600">

*An example of the workflow of the `mfe` pipeline*

### 1. Set-up

- Node `init_data` (interface `InitData` in `interfaces/data.py`): defines the paths to all data
files of the subject, assuming the default organisation of directories from HCP
- Node `save_features` (interface `SaveFeatures` in `interfaces/data.py`): vectorise and saves 
extracted features to a `HDF5` file for storage

### 2. Psychometric and confounding variables

- Node `pheno` (interface `Phenotypes` in `interfaces/data.py`): extracts all psychometric variables
based on the configurations defined by `dataset_params` in `utilities.py`
- Node `conf` (interface `Confounds` in `interfaces/features.py`): extracts the eight confounding
variables

### 3. Functionl MRI features

- Node `rsfc` (interface `FC` in `interfaces/features.py`): applys nuisance regression to the 
resting-state time-series, parcellate the with four levels of matching Schaefer + Melbourne atlas, 
and compute the model-free/static and dynamic functional connectivity. If diffusion data are also 
used, effective connectivity is also computed using structural connectivity as prior.
- Node `rs_stats` (interface `NetworkStats` in `interfaces/features.py`): computes network 
statisitcs based on the model-free resting-state functional connectivity
- Node `tfc` (interface `FC` in `interfaces/features.py`): same as `rsfc` but for task MRI data

### 4. Structural MRI features

- Node `anat` (interface `Anat` in `interfaces/features.py`): extracts the myelin estmate, cortical 
surface area, cortical thickness, and gray matter volume

### 5. Diffusion MRI features

- Node `sub_dir` (interface `AddSubDir` in `utilities.py`): creates a subject directory for
`FreeSurfer`
- Node `pick_atlas` (interface `PickAtlas` in `interfaces/data.py`): parses atlases of each level
- Node `add_annot` (interface `SubDirAnnot` in `interaces/data.py`): add Schaefer atlas annotation 
files to the `FreeSurfer` subject directory
- Node `aseg` (interface `CombineStrings` in `utilities.py`): defines file name for the 
aparc-to-aseg transform

#### 5.1. Transform the Melbourne atlas to the subject T1 space

- Node `std2t1` (interface `InvWarp` from `nipype.interfaces.fsl`): inverts the T1-to-MNI transform 
in the HCP data for a MNI-to-T1 transform
- Node `mel_t1` (interface `ApplyWarp` from `nipype.interfaces.fsl`): transforms the Melbourne 
atlas to subject T1 space using the MNI-to-T1 transform

#### 5.2. Transform the Schaefer atlas to the subject T1 space

- Node `lannot_sub` (interface `SurfaceTransform` from `nipype.interfaces.freesurfer`): transforms 
the Schaefer atlas in the left hemisphere to the subject native surface space
- Node `rannot_sub` (interface `SurfaceTransform` from `nipype.interfaces.freesurfer`): same as 
`lannot_sub` but for the right hemispehre
- Node `sch_aseg` (interface `Aparc2Aseg` from `nipype.interfaces.freesurfer`): transforms the 
Schaefer atlas to the subject volumetric segmentation space
- Node `sch_t1` (interface `FLIRT` from `nipype.interfaces.fsl`): transforms the Schaefer atlas 
to the subject T1 space

#### 5.3. Combine the two atlases

- Node `combine` (interface `CombineAtlas` in `utilities.py`): combine the Schaefer and the 
Melbourne atlas into one file
- Node `downsamp` (interface `FLIRT` from `nipype.interfaces.fsl`): downsample the combined atlas 
to the same resolution as the diffusion data

#### 5.4. Tractography and structural connectome

- Node `prob_track` (interface `ProbTract` in `interfaces/diffusion.py`): estimates fiber 
orientation and computes tractography using `MRTrix3`
- Node `sc` (interface `SC` in `interfaces/features.py`): computes structural connectome based on 
streamline count and length-scaled count using `MRTrix3`

## DTI feature extraction (`mfe_dti`)

The workflow is defined in `dti.py`.

- Node `dtifit` (interface `DTIFit` in `interfaces/diffusion.py`): computes the DTI images for 
each subject
- Node `tbss` (interface `TBSS` in `interfaces/diffusion.py): create dataset-wide skeleton maps 
based on the FA images from all subjects
- Node `features` (interface `DTIFeatures` in `interfaces/features.py`): parcellate the 
skeletonised DTI images using the JHU white matter atlas, and extract the DTI-based features
- Node `save_features` (interface `SaveDTIFeatures` in `interfaces/data.py`): saves the DTI-based 
features from all subjects to one `HDF5` file for storage


