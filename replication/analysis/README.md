This folder contains analysis scripts for the prediction outcomes. Each analysis is explained below.
With the repository placed inside a `${project_dir}`, the following commands refer to this current 
folder as:

```bash
analysis_dir=${project_dir}/mpp/replication/analysis
```

## 1. Main analyses (figures 1-4, S2)

The main analyses and figures in the manuscript are performed and plotted respectively using the 
two scripts, `collect_results.py` and `plot_figures.py`. This includes analyses for:

- the trend of prediction performance over increasing levels of multimodal integration
- finding the "best" and "necessary" feature sets
- determining the useful feature types based on "useful" predictive models

```bash
# Collect prediction results into tables
python3 ${analysis_dir}/collect_results.py --datasets HCP-A HCP-YA HCP-D \
    --pred_dir ${project_dir}/mpp_output --out_dir ${analysis_dir}/analysis_results

# Plot all figures
python3 ${analysis_dir}/plot_figures.py --res_dir ${analysis_dir}/analysis_results \
    --out_dir ${analysis_dir}/figures_raw
```

## 2. Sensitivity analyses

Three sensitivity analyses were added during revision.

### 2.1. Alternative model (figures S5-6)

An alternative prediction pipeline is implemented using support vector regression (SVR) at the 
feature-wise level instead. The alternative model is incorporated in the main analysis code, 
triggered by the option `--model alternative`. We ran the alternative pipeline for the "useful" 
models in HCP-D and HCP-A:

```bash
# HCP-D
mpp --datasets HCP-D --targets totalcogcomp --features_dir ${project_dir}/mfe_output \
    --sublists ${project_dir}/sublist/HCP-D_allRun.csv --level 3 \
    --work_dir ${project_dir}/work --output_dir ${project_dir}/mpp_output/HCP-D --model alternative

# HCP-A
mpp --datasets HCP-A --targets totalcogcomp crycogcomp fluidcogcomp reading picvocab \
    --features_dir ${project_dir}/mfe_output \
    --sublists ${project_dir}/sublist/HCP-A_allRun.csv --level 3 \
    --work_dir ${project_dir}/work --output_dir ${project_dir}/mpp_output/HCP-A --model alternative
```

The prediction outcomes are then analysed with the two scripts in `revision_analysis`:

```bash
# Collect prediction results into tables
for dataset in HCP-D HCP-YA HCP-A; do 
    python3 ${analysis_dir}/revision_analysis/collect_alter_results.py --dataset ${dataset} \
        --pred_dir ${project_dir}/mpp_output --out_dir ${analysis_dir}/analysis_results
done

# Plot all figures
python3 ${analysis_dir}/revision_analysis/plot_alter_figures.py \
    --res_dir ${analysis_dir}/analysis_results --out_dir ${analysis_dir}/figures_raw
```

### 2.2. Fusion CCA analysis (figures S?)

We analysed the feature importance sensitivity in a CCA-based fusion model. The scripts used and 
the results can be found in `revision_analysis/fusion_cca_analysis`.

```bash
fusion_analysis_dir=${analysis_dir}/revision_analysis/fusion_cca_analysis

# Extract all brain and phenotypic data
for dataset in HCP-D HCP-YA HCP-A; do 
    python3 ${fusion_analysis_dir}/collect_data_fusion.py --dataset ${dataset} \
        --sublist_dir ${project_dir}/sublist --out_dir ${project_dir}/fusion_cca_data \
        --work_dir ${project_dir}/work
done

# Generate indices for cross-validation
for dataset in HCP-D HCP-YA HCP-A; do 
    python3 ${fusion_analysis_dir}/compute_cv_indices.py --dataset ${dataset} \
        --data_dir ${project_dir}/fusion_cca_data --out_dir ${project_dir}/fusion_cca_data \
        --hcpya_res ${project_dir}/phenotype/restricted_hcpya.csv
done

# Compute cross-validation-specific features, i.e., gradient and structural co-registration features
for dataset in HCP-D HCP-YA HCP-A; do
    for fold in $(seq 0 99); do
        python3 ${fusion_analysis_dir}/compute_cv_features.py --dataset ${dataset} --fold ${fold} \
            --data_dir ${project_dir}/fusion_cca_data --out_dir ${project_dir}/fusion_cca_data \
    done
done

# Run CCA-based fusion models separately for connectivity and region-wise features
for dataset in HCP-D HCP-YA HCP-A; do 
    python3 ${fusion_analysis_dir}/fusion_cca_conn.py --dataset ${dataset} \
        --data_dir ${project_dir}/fusion_cca_data --out_dir ${project_dir}/fusion_cca_results
    python3 ${fusion_analysis_dir}/fusion_cca_region.py --dataset ${dataset} \
        --data_dir ${project_dir}/fusion_cca_data --out_dir ${project_dir}/fusion_cca_results
done
```

Due to the large amount and size, the data and results are saved to a DataLad dataset. The files 
are then retrieved and dropped one by one to extract the relevant results:

```bash
python3 ${fusion_analysis_dir}/collect_fusion_cca_results.py \
    --out_dir ${project_dir}/fusion_cca_results --work_dir ${project_dir}/work
```

Finally, plot the results:

### 2.3. Fusion PCA analysis (figures S?)

To examine the sensitivity of main analysis outcomes to feature dimensionality, we added a model 
where dimensionality reduction via PCA, preserving 95% of the variance, is performed before 
feature-wise and integrated predictions. The PCA-based model is incorporated in the main analysis 
code, represented by the option `--model pca-en`. We ran the PCA-based fusion model for the 
prediction of total cognition composite score in HCP-D and HCP-A:

```bash
# HCP-D
mpp --datasets HCP-D --targets totalcogcomp --features_dir ${project_dir}/mfe_output \
    --sublists ${project_dir}/sublist/HCP-D_allRun.csv --level 3 \
    --work_dir ${project_dir}/work --output_dir ${project_dir}/mpp_output/HCP-D --model pca-en

# HCP-A
mpp --datasets HCP-A --targets totalcogcomp --features_dir ${project_dir}/mfe_output \
    --sublists ${project_dir}/sublist/HCP-A_allRun.csv --level 3 \
    --work_dir ${project_dir}/work --output_dir ${project_dir}/mpp_output/HCP-A --model pca-en
```

## 3. Supplementary analyses

### 3.1. Correlation between psychometric variables

We plotted the correlation between all pairs of psychometric variables, averaged across datasets. 

```bash
python3 ${analysis_dir}/revision_analysis/pheno_corr.py --data_dir ${project_dir}/mfe_output \
    --sublist_dir ${project_dir}/sublist --out_dir ${analysis_dir}/figures_raw
```

### 3.2. Summary of head motion (figure S3)

In each dataset, framewise displacement (FD) was computed using `FSL` based on denoised timeseries. 
Then, we plotted the distribution of FD values in each dataset, as well as its correlation with 
every psychometric variable in each dataset.

```bash
python3 ${analysis_dir}/revision_analysis/plot_fd.py --data_dir ${project_dir}/fd_data \
    --out_dir ${analysis_dir}/figures_raw
```

### 3.3 Confound model (figure S1)

For each psychometric variable in each dataset, a confound model was also built to predict the 
psychometric variable using confounding variables that were controlled in the main analysis. We 
plotted the prediction accuracies of these confound models to compare against the brain-based 
models.

```bash
python3 ${analysis_dir}/revision_analysis/conf_results --res_dir ${project_dir}/mpp_output \
    --out_dir ${analysis_dir}/figures_raw
```

### 3.4. Interindividual variability of cortical surface area (figure S4)

To assess whether the importance of cortical surface area (SA) in HCP-A is due to higher 
interindividual variability in the cohort, we computed the interindividual variability of SA in 
all three cohorts. Then, we plotted the distribution of the variability in each cohort.

```bash
python3 ${analysis_dir}/revision_analysis/plot_sa.py --sublist_dir ${project_dir}/sublist \
    --data_out_dir ${analysis_dir}/analysis_results --plot_out_dir ${analysis_dir}/figures_raw \
    --work_dir ${project_dir}/work
```
