# Multimodal brain-based prediction pipeline (`mpp`)

The pipeline `mpp` runs stacked prediction of psychometric variables using multimodal brain 
features. The workflow is defined in `main.py`.

<img src="mpp_graph.png" height="400">

*An example of the workflow of the prediction pipeline*

## 1. Set-up

- Node `sublsit` (interface `PredictSublist` from `interfaces/data.py`): generates a sublist for 
each psychometric variable to predict, to remove subjects with missing data
- Node `cv_split` (interface `CrossValSplit` from `interfaces/crossval.py`): generates the 
cross-validation indices for outer and inner/nested folds
- Node `features` (interface `CVFeatures` in `interfaces/features.py`): computes the diffusion 
mapping embedding and structural co-registration parameters using the training data in each fold

## 2. Feature-wise models (feature-level)

- Node `fw_model` (interface `FeaturewiseModel` in `interfaces/crossval.py`): implements a 
feature-wise model for each feature type. Gradient embeddings and structural co-registration 
parameters are applied to training and test data to generate these cross-validation sensative 
features. A 5-fold nested cross-validation loop is used to generate the feature-wise model 
predictions in the outer loop training data, to provide as features to the meta-level model
- Node `fw_combine` (interface `PredictionCombine` in `interfaces/data.py`): combines predictions 
from the different feature-wise models for saving
- Node `fw_save` (interface `PredictionSave` in `interfaces/data.py`): saves prediction outcomes 
of the feature-wise models

## 3. Confound model

- Node `conf_model` (interface `ConfoundModel` in `interfaces/crossval.py`): implements a confound 
model to predict the psychometric variable using confounding variables
- Node `conf_save` (interface `PredictionSave` in `interfaces/data.py`): saves prediction outcomes 
of the confound model

## 4. Integrated-features set models (meta-level)

- Node `if_model` (interface `IntegratedFeaturesModel` in `interfaces/crossval.py`): implements 
integrated-features set models with increasing levels of multimodal integration
- Node `if_save` (interface `PredictionSave` in `interfaces/data.py`): saves prediction outcomes of 
the integrated-features set models

## 5. Other notable functions

- `pheno_reg_conf` in `utilities.py`: confound regression used in feature-level and meta-level 
models. Regression parameters are estimated in the training data and applied to the test data.
- `elastic_net` in `utilities.py`: elastic net implementation using the recommended L1 ratio 
parameter space from Scikit-learn, used in feature-level mdoels for the default pipeline
- `linear_svr` in `utilities.py`: linear SVR implementation using default options, used in 
feature-level models for the alternative pipeline
- `_random_forest_cv` in `IntegratedFeaturesModel` (in `interfaces/crossval.py`): Random Forest 
regression implementation using grid search for hyperparameter optimisation, used in meta-level 
models
- `_train_ranks` in `IntegratedFeaturesModel` (in `interfaces/crossval.py`): estimation of feature 
ranks based on training set prediction performance, used in meta-level models


