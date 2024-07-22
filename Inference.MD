# Takoi Inference Method
This section describes the inference method for the test.csv file found at the URL below. If you want to perform inference on other data, please replace test.csv with the other data..</br>
https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data

The trained model has been shared with the competition host (https://www.kaggle.com/jerrylin96) at the following dataset:
https://www.kaggle.com/datasets/takoihiraokazu/takoi-leap-output


## Folder
Go to `./solution`

## Environment

```
docker-compose up --build
```

## Data
- Download the following data to `./storage/leap/data` </br>
https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data

## Placing Preprocessing Files
- If the directory `./storage/leap/output/fe` does not exist, please create it.

From the following dataset, place the specified folders into the designated directory:</br>
https://www.kaggle.com/datasets/takoihiraokazu/takoi-leap-output
- fe101 : `./storage/leap/output/fe`
- fe102 : `./storage/leap/output/fe`
- fe103 : `./storage/leap/output/fe`

## Placing Trained Models
- If the directory `./storage/leap/output/exp` does not exist, please create it.

From the following dataset, place the specified folders into the designated directory:</br>
https://www.kaggle.com/datasets/takoihiraokazu/takoi-leap-output
- ex124 : `./storage/leap/output/exp`
- ex130 : `./storage/leap/output/exp`
- ex131 : `./storage/leap/output/exp`
- ex133 : `./storage/leap/output/exp`
- ex133 : `./storage/leap/output/exp`
- ex134 : `./storage/leap/output/exp`
- ex135 : `./storage/leap/output/exp`
- ex136 : `./storage/leap/output/exp`
- ex138 : `./storage/leap/output/exp`
- ex139 : `./storage/leap/output/exp`
- ex141 : `./storage/leap/output/exp`
- ex159 : `./storage/leap/output/exp`
- ex162 : `./storage/leap/output/exp`

For the inference of the test data, the following two files are required for ex124 (and similarly for other experiment numbers):

- ex124/exp124_clapse_col.npy
    - Contains targets with low prediction accuracy (R2 below 0). Used to set their prediction results to 0.
- ex124/model/ex124.pth
    - Trained model

The specified folders already contain prediction results for the test data(e.g., ex124_pp.parquet), so if you perform a new inference, please overwrite the existing prediction results.


## Test Data Preprocessing
Please run the following files in `./fe` </br>
- fe100_data_save_new_test.ipynb
    - Converts test.csv to test.parquet
- fe134_make_seq_test.ipynb
    - Creates features
- fe135_make_seq_diff_test.ipynb
    - Creates features
- fe136_make_state_sum_test.ipynb
    - Creates features

## Test Data Inference
Please run the following files in the solution directory
- exp124_test_pp.py
- exp130_test_pp.py
- exp131_test_pp.py
- exp133_test_pp.py
- exp134_test_pp.py
- exp135_test_pp.py
- exp136_test_pp.py
- exp138_test_pp.py
- exp139_test_pp.py
- exp141_test_pp.py
- exp159_test_pp.py
- exp162_test_pp.py

The prediction result file for ex124 will be:
`./storage/leap/output/exp/ex124/exp124_pp.parquet`
For other experiment numbers, replace 124 with the respective experiment number in the file name.

## Ensemble
Before running the ensemble, please copy the prediction results from each teammate into the directories below. Create the directories if they do not exist.

The prediction results to be used are listed under each directory.

takoi :`./storage/leap/output/exp/takoi_pred3`
- ex130_pp.parquet
- ex131_pp.parquet
- ex133_pp.parquet
- ex134_pp.parquet
- ex135_pp.parquet
- ex136_pp.parquet
- ex138_pp.parquet
- ex139_pp.parquet
- ex141_pp.parquet
- ex159_pp.parquet
- ex162_pp.parquet

kami : `./storage/leap/output/exp/ensemble0714/kami-leap-pred2`
- kami_experiments_201_unet_multi_all_384_n2_submission.parquet
- kami_experiments_201_unet_multi_all_512_n3_submission.parquet
- kami_experiments_201_unet_multi_all_n3_restart2_submission.parquet
- kami_experiments_201_unet_multi_all_submission.parquet
- kami_experiments_204_diff_last_all_lr_submission.parquet
- kami_experiments_217_fix_transformer_leak_all_cos_head64_submission.parquet
- kami_experiments_217_fix_transformer_leak_all_cos_head64_n4_submission.parquet
- kami_experiments_222_wo_transformer_all_submission.parquet
- kami_experiments_222_wo_transformer_all_004_submission.parquet
- kami_experiments_225_smoothl1_loss_all_005_submission.parquet
- kami_experiments_225_smoothl1_loss_all_beta_submission.parquet

kurupical : `./storage/leap/output/exp/ensemble0715/kurupical-leap-pred2`
- 20240703230157_exp042_70m_transformer_512x4_lr0.001_beta1/submission.parquet
- 20240705215850_exp042_70m_transformer_768x4_lr0.001_beta1/submission.parquet
- 20240706022848_exp042_70m_cnn64_smoothl1beta1_lr2.5e-3_beta0.01_wd0.05/submission.parquet
- 20240708233043_exp042_70m_cnn96_smoothl1beta1_lr2e-3_beta0.01_wd0.05/submission.parquet
- 20240709224049_exp042_70m_cnn128_smoothl1beta1_lr2e-3_beta0.01_wd0.05/submission.parquet
- 20240713043714_exp042_70m_cnn160_smoothl1beta1_lr2e-3_beta0.01_wd0.05_ddp/submission.parquet
- 20240714093820_exp042_70m_cnn160_smoothl1beta1_lr2e-3_beta0.01_wd0.05_ddp/submission.parquet

Please run the following file
- exp173_team_ensemble_onlyt_test.py
    - The output file 20240715_ensemble_per_target_500.parquet will be used to overwrite the ensemble results of teammate kurupical.


