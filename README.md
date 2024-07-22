# LEAP - Atmospheric Physics using AI (ClimSim)
This is the takoi part of the 4th place solution for LEAP - Atmospheric Physics using AI (ClimSim).</br>

The README includes the procedures for training, evaluation, and inference on test data. If you only want to perform inference on test data, please refer to Inference.MD.

## Data Preparation

Go to data_preparation </r>

This directory is used for processing Huggingface LEAP/ClimSim_low-res data on a monthly basis (e.g., 0001-2, 0001-3, etc.). The generated data files are named following the pattern train_year_month.parquet, such as train_0001_2.parquet. Additionally, the generated data is organized by year into separate directories. The directory names follow the pattern train_year, such as train_0001 or train_0002.

Finally, move the `./storage` directory to the `solution` directory.


If Huggingface LEAP/ClimSim_low-res data has already been processed and organized monthly, processing in this directory is not required. In that case, please move the data to the directory mentioned above.

### Environment
```
docker-compose up --build
```

### Data download
Please run the following notebooks in ./data_load

### Make data
Please run the notebook make_data_yyyy_m.ipynb located in the data_preparation directory.

### Move directory
move the `./storage` directory to the `solution` directory.

## Solution
Go to solution </r>

### Data download
Download data to ./storage/leap/data/ from https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data .

### Environment
```
docker-compose up --build
```
### Hardware
GPU: A100 (80GB)</br>
Memory: 240GB (The memory usage was adjusted according to the code being executed, but for this competition, the maximum environment used was 240GB.)


### Feature Engineering

Please run the following notebooks in ./fe </br>
Due to the large data size, the data has been split for processing, resulting in a larger number of notebooks.The following notebooks are separated by purpose, but please execute the code in the order of the experiment numbers (the *** part of fe***).

- Preprocess Kaggle Data </br>
The following notebooks include calculations such as standardization using StandardScaler.
    - fe100_data_save_new_test.ipynb
    - fe101_data_preprocess.ipynb
    - fe102_data_preprocess_diff.ipynb
    - fe103_data_preprocess_sum_feature.ipynb
    - fe129_data_save.ipynb

- Preprocess LEAP/ClimSim_low-res </br>
The following notebooks handle processes such as merging monthly data
    - fe055_data_save_8.ipynb
    - fe075_data_save_6_7.ipynb
    - fe081_data_save_5.ipynb
    - fe087_data_save_4.ipynb
    - fe092_data_save_3.ipynb
    - fe104_data_save.ipynb
    - fe108_data_save.ipynb
    - fe112_data_save.ipynb
    - fe116_data_save_2.ipynb
    - fe117_data_save.ipynb
    - fe121_data_save_1.ipynb
    - fe122_data_save.ipynb
    - fe137_data_save_8_1.ipynb
    - fe141_data_save_6_7_1.ipynb
    - fe159_data_save_4_5_1.ipynb
    - fe164_data_save_2_3_1.ipynb
    - fe168_data_save_1_1.ipynb

- Make features(Train)</br>
The following notebooks create sequences for training
    - fe105_make_seq_104.ipynb
    - fe106_make_seq_diff_104.ipynb
    - fe107_make_state_sum_104.ipynb
    - fe109_make_seq_108.ipynb
    - fe110_make_seq_diff_108.ipynb
    - fe111_make_state_sum_108.ipynb
    - fe113_make_seq_112.ipynb
    - fe114_make_seq_diff_112.ipynb
    - fe115_make_state_sum_112.ipynb
    - fe118_make_seq_117.ipynb
    - fe119_make_seq_diff_117.ipynb
    - fe120_make_state_sum_117.ipynb
    - fe123_make_seq_122.ipynb
    - fe124_make_seq_diff_122.ipynb
    - fe125_make_state_sum_122.ipynb
    - fe138_make_seq_137.ipynb
    - fe139_make_seq_diff_137.ipynb
    - fe140_make_state_sum_137.ipynb
    - fe142_make_seq_141.ipynb
    - fe143_make_seq_diff_141.ipynb
    - fe144_make_state_sum_141.ipynb
    - fe160_make_seq_159.ipynb
    - fe161_make_seq_diff_159.ipynb
    - fe162_make_state_sum_159.ipynb
    - fe165_make_seq_164.ipynb
    - fe166_make_seq_diff_164.ipynb
    - fe167_make_state_sum_164.ipynb
    - fe169_make_seq_168.ipynb
    - fe170_make_seq_diff_168.ipynb
    - fe171_make_state_sum_168.ipynb
   
- Save in segments</br>
The following notebooks save the sequences created in Make Features (Train) in separate segments
    - fe126_data_save_1_3_118_123.ipynb
    - fe127_data_save_4_5_113.ipynb
    - fe128_data_save_6_8_105_109.ipynb
    - fe145_data_save_6_8_138_142_1.ipynb
    - fe163_data_save_4_5_160_1.ipynb
    - fe172_data_save_1_3_165_169_1.ipynb


- Make features(Validation)</br>
The following notebooks create sequences for validation
    - fe130_make_seq_129.ipynb
    - fe131_make_seq_diff_129.ipynb
    - fe132_make_state_sum_129.ipynb

- Make features(Test)</br>
The following notebooks create sequences for test
    - fe134_make_seq_test.ipynb
    - fe135_make_seq_diff_test.ipynb
    - fe136_make_state_sum_test.ipynb

### Training & Validation
Please run the following script </br>
- Train
    - exp124.py
    - exp130.py
    - exp131.py
    - exp133.py
    - exp134.py
    - exp135.py
    - exp136.py
    - exp138.py
    - exp139.py
    - exp141.py
    - exp159.py
    - exp162.py
- Validation
    - exp124_val.py
    - exp130_val.py
    - exp131_val.py
    - exp133_val.py
    - exp134_val.py
    - exp135_val.py
    - exp136_val.py
    - exp138_val.py
    - exp139_val.py
    - exp141_val.py
    - exp159_val.py
    - exp162_val.py
### Test inference
Please run the following script </br> 
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

### Ensemble
Please copy the prediction files with the format exp***_val_preds.npy and ex***_pp.parquet from Validation and Test inference to storage/leap/output/exp/takoi-pred3. Also, copy teammate kami's prediction results to storage/leap/output/exp/ensemble0714/kami-leap-pred2, and kurupical's prediction results to storage/leap/output/exp/ensemble0715/kurupical-leap-pred2.
Please run the following script </br>
- exp173_team_ensemble.py
- exp173_team_ensemble_test.py
    - The w_list in exp173_team_ensemble_test.py is the w_list output by exp173_team_ensemble.py. </br>
    
The results from exp173_team_ensemble_test.py will be used to overwrite the stacking prediction results conducted by teammate kurupical.






