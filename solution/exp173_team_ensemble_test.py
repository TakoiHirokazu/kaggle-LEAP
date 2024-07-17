import logging
from contextlib import contextmanager
import time
import sys
from pathlib import Path
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tqdm
from sklearn.metrics import r2_score
from scipy.optimize import minimize

# =========================
# Settings
# =========================
DATA_DIR = Path("storage/leap/data/")
OUTPUT_DIR = Path("storage/leap/output")
exp = "173"
exp_dir = OUTPUT_DIR / "exp" / f"ex{exp}"
model_dir = exp_dir / "model"
logger_path = exp_dir / f"ex{exp}.txt"
exp_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
ENSEMBLE_DIR = "storage/leap/output/exp/ensemble0714"
ENSEMBLE_DIR2 = "storage/leap/output/exp/ensemble0715"
LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

max_iter = 500


def setup_logger(out_file=None, stderr=True,
                 stderr_level=logging.INFO, file_level=logging.DEBUG):
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER


def read_inference(file):
    if ".npy" in file:
        ary = np.load(file)
        if debug:
            ary = ary[::10]
        ary = ary / weight.reshape(1, -1)
        ary[np.isnan(ary)] = 0
    elif ".parquet" in file:
        ary = pd.read_parquet(file)[TARGET_COLUMNS]
        if debug:
            ary = ary[::10]
    assert ary.shape == (len(label), 368)
    return ary


def r2(weights, pred, label):
    score = r2_score(label, (pred * weights.reshape(1, 1, -1)).sum(axis=2))
    return score


def objective_function(weights, predictions, true_labels):
    """Objective function to minimize (negative accuracy or error).

    Args:
        weights (np.array): Weights for the individual models.
        predictions (list of np.array): Predictions from individual models.
        true_labels (np.array): True labels.

    Returns:
        float: Objective value (e.g., negative accuracy).
    """
    score = r2(weights, predictions, true_labels)
    # print(score, weights)
    return -score


def nelder_mead_optimization(predictions, true_labels, max_iterations=250):
    """Optimize ensemble weights using Nelder-Mead method.

    Args:
        predictions (list of np.array): Predictions from individual models.
        true_labels (np.array): True labels.
        max_iterations (int): Maximum number of iterations.

    Returns:
        dict: Results of optimization including final weights and objective value.
    """
    initial_weights = weights[:]
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})

    result = minimize(objective_function, initial_weights,
                      args=(predictions, true_labels),
                      method='Nelder-Mead',
                      options={'maxiter': max_iterations}, constraints=constraints)

    final_weights = result.x
    final_score = result.fun

    return {'weights': final_weights, 'error': final_score}


@ contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s')


setup_logger(out_file=logger_path)

debug = False

TARGET_COLUMNS = pd.read_csv(
    "storage/leap/data/sample_submission.csv").columns[1:]
assert len(TARGET_COLUMNS) == 368
label = pd.read_parquet(OUTPUT_DIR / "fe" / "fe129" /
                        "fe129_train.parquet")[TARGET_COLUMNS].to_numpy()
weight = pd.read_csv("storage/leap/data/sample_submission.csv",
                     nrows=1).values[0, 1:].astype(np.float32)
n_data_for_eval = 368 * 1730
label = label[-n_data_for_eval:]

preds_val = []
preds_test = []
exp_names = []

# ====================
# kurupical submissions
# ====================
#
# 　  　 ／) ／)
# 　 　/　⌒　ヽ
# 　 ｜●_ ● |/＼
# 　 (〇 ～ 〇| ／
# 　 /　　　　|<
# 　｜　　 L/ |/
#

kurupical_folders = [
    # "/kaggle/input/kurupical-leap-pred2/20240621215730_exp031_20files_1d_dims(64, 128, 256, 512)_7epochs_power2.5e-3",
    ENSEMBLE_DIR2 + "/kurupical-leap-pred2/20240703230157_exp042_70m_transformer_512x4_lr0.001_beta1",
    ENSEMBLE_DIR2 + "/kurupical-leap-pred2/20240705215850_exp042_70m_transformer_768x4_lr0.001_beta1",
    ENSEMBLE_DIR2 + "/kurupical-leap-pred2/20240706022848_exp042_70m_cnn64_smoothl1beta1_lr2.5e-3_beta0.01_wd0.05",
    ENSEMBLE_DIR2 + "/kurupical-leap-pred2/20240708233043_exp042_70m_cnn96_smoothl1beta1_lr2e-3_beta0.01_wd0.05",
    ENSEMBLE_DIR2 + "/kurupical-leap-pred2/20240709224049_exp042_70m_cnn128_smoothl1beta1_lr2e-3_beta0.01_wd0.05",
    ENSEMBLE_DIR2 + "/kurupical-leap-pred2/20240713043714_exp042_70m_cnn160_smoothl1beta1_lr2e-3_beta0.01_wd0.05_ddp",
    ENSEMBLE_DIR2 + "/kurupical-leap-pred2/20240714093820_exp042_70m_cnn160_smoothl1beta1_lr2e-3_beta0.01_wd0.05_ddp",
]

for folder in tqdm.tqdm(kurupical_folders):
    pred_test = pd.read_parquet(
        f"{folder}/submission.parquet")[TARGET_COLUMNS].values.astype(np.float64)

    # print(folder, pred_val.shape, pred_test.shape, pred_val.dtype, pred_test.dtype)
    preds_test.append(pred_test)
    exp_names.append(folder)


# takoi_files = glob.glob("/kaggle/input/leap-takoi-pred2/*_val_preds.npy")
takoi_files = [
    # "/kaggle/input/leap-takoi-pred3/exp123_val_preds.npy",
    # "/kaggle/input/leap-takoi-pred3/exp124_val_preds.npy",
    "storage/leap/output/exp/takoi_pred3/exp130_val_preds.npy",
    "storage/leap/output/exp/takoi_pred3/exp131_val_preds.npy",
    "storage/leap/output/exp/takoi_pred3/exp133_val_preds.npy",
    "storage/leap/output/exp/takoi_pred3/exp134_val_preds.npy",
    "storage/leap/output/exp/takoi_pred3/exp135_val_preds.npy",
    "storage/leap/output/exp/takoi_pred3/exp136_val_preds.npy",
    "storage/leap/output/exp/takoi_pred3/exp138_val_preds.npy",
    "storage/leap/output/exp/takoi_pred3/exp139_val_preds.npy",
    "storage/leap/output/exp/takoi_pred3/exp141_val_preds.npy",
    "storage/leap/output/exp/takoi_pred3/exp159_val_preds.npy",
    "storage/leap/output/exp/takoi_pred3/exp162_val_preds.npy",
]
takoi_files_test = [
    # "/kaggle/input/leap-takoi-pred3/exp123_val_preds.npy",
    # "/kaggle/input/leap-takoi-pred3/exp124_val_preds.npy",
    "storage/leap/output/exp/takoi_pred3/ex130_pp.parquet",
    "storage/leap/output/exp/takoi_pred3/ex131_pp.parquet",
    "storage/leap/output/exp/takoi_pred3/ex133_pp.parquet",
    "storage/leap/output/exp/takoi_pred3/ex134_pp.parquet",
    "storage/leap/output/exp/takoi_pred3/ex135_pp.parquet",
    "storage/leap/output/exp/takoi_pred3/ex136_pp.parquet",
    "storage/leap/output/exp/takoi_pred3/ex138_pp.parquet",
    "storage/leap/output/exp/takoi_pred3/ex139_pp.parquet",
    "storage/leap/output/exp/takoi_pred3/ex141_pp.parquet",
    "storage/leap/output/exp/takoi_pred3/ex159_pp.parquet",
    "storage/leap/output/exp/takoi_pred3/ex162_pp.parquet",
]


# ====================
# takoi submissions
# ====================


for n, file in tqdm.tqdm(enumerate(takoi_files)):
    # pred_val = np.load(file)[-n_data_for_eval:]
    # pred_val = pred_val / weight.reshape(1, -1)
    # pred_val[np.isnan(pred_val)] = 0

    pred_test = pd.read_parquet(takoi_files_test[n])[TARGET_COLUMNS].values
    preds_test.append(pred_test)
    exp_names.append(file)


# ====================
# kami submissions
# ===================
kami_files = [
    ENSEMBLE_DIR + "/kami-leap-pred2/kami_experiments_201_unet_multi_all_384_n2_valid_pred.parquet",
    ENSEMBLE_DIR + "/kami-leap-pred2/kami_experiments_201_unet_multi_all_512_n3_valid_pred.parquet",
    ENSEMBLE_DIR + "/kami-leap-pred2/kami_experiments_201_unet_multi_all_n3_restart2_valid_pred.parquet",
    ENSEMBLE_DIR + "/kami-leap-pred2/kami_experiments_201_unet_multi_all_valid_pred.parquet",
    ENSEMBLE_DIR + "/kami-leap-pred2/kami_experiments_204_diff_last_all_lr_valid_pred.parquet",
    # "/kaggle/input/kami-leap-pred2/kami_experiments_211_simple_split_head_all_cos_valid_pred.parquet",
    ENSEMBLE_DIR + "/kami-leap-pred2/kami_experiments_217_fix_transformer_leak_all_cos_head64_valid_pred.parquet",
    ENSEMBLE_DIR + "/kami-leap-pred2/kami_experiments_217_fix_transformer_leak_all_cos_head64_n4_valid_pred.parquet",
    ENSEMBLE_DIR + "/kami-leap-pred2/kami_experiments_222_wo_transformer_all_valid_pred.parquet",
    ENSEMBLE_DIR + "/kami-leap-pred2/kami_experiments_222_wo_transformer_all_004_valid_pred.parquet",
    ENSEMBLE_DIR + "/kami-leap-pred2/kami_experiments_225_smoothl1_loss_all_005_valid_pred.parquet",
    ENSEMBLE_DIR + "/kami-leap-pred2/kami_experiments_225_smoothl1_loss_all_beta_valid_pred.parquet"

]

for file in tqdm.tqdm(kami_files):

    pred_test = pd.read_parquet(file.replace(
        "_valid_pred.parquet", "_submission.parquet"))[TARGET_COLUMNS].values

    preds_test.append(pred_test)
    exp_names.append(file)


ALL_ZERO_TARGET_COLS = [
    "ptend_q0001_0",
    "ptend_q0001_1",
    "ptend_q0001_2",
    "ptend_q0001_3",
    "ptend_q0001_4",
    "ptend_q0001_5",
    "ptend_q0001_6",
    "ptend_q0001_7",
    "ptend_q0001_8",
    "ptend_q0001_9",
    "ptend_q0001_10",
    "ptend_q0001_11",
    "ptend_q0002_0",
    "ptend_q0002_1",
    "ptend_q0002_2",
    "ptend_q0002_3",
    "ptend_q0002_4",
    "ptend_q0002_5",
    "ptend_q0002_6",
    "ptend_q0002_7",
    "ptend_q0002_8",
    "ptend_q0002_9",
    "ptend_q0002_10",
    "ptend_q0002_11",
    "ptend_q0003_0",
    "ptend_q0003_1",
    "ptend_q0003_2",
    "ptend_q0003_3",
    "ptend_q0003_4",
    "ptend_q0003_5",
    "ptend_q0003_6",
    "ptend_q0003_7",
    "ptend_q0003_8",
    "ptend_q0003_9",
    "ptend_q0003_10",
    "ptend_q0003_11",
    "ptend_q0002_12",
    "ptend_q0002_13",
    "ptend_q0002_14",
    "ptend_q0002_15",
    "ptend_q0002_16",
    "ptend_q0002_17",
    "ptend_q0002_18",
    "ptend_q0002_19",
    "ptend_q0002_20",
    "ptend_q0002_21",
    "ptend_q0002_22",
    "ptend_q0002_23",
    "ptend_q0002_24",
    "ptend_q0002_25",
    "ptend_q0002_26",
    "ptend_q0002_27",
    "ptend_u_0",
    "ptend_u_1",
    "ptend_u_2",
    "ptend_u_3",
    "ptend_u_4",
    "ptend_u_5",
    "ptend_u_6",
    "ptend_u_7",
    "ptend_u_8",
    "ptend_u_9",
    "ptend_u_10",
    "ptend_u_11",
    "ptend_v_0",
    "ptend_v_1",
    "ptend_v_2",
    "ptend_v_3",
    "ptend_v_4",
    "ptend_v_5",
    "ptend_v_6",
    "ptend_v_7",
    "ptend_v_8",
    "ptend_v_9",
    "ptend_v_10",
    "ptend_v_11",
]


w_list = [[-0.10952915,  0.07556821,  0.05311548,  0.28646586,  0.16097361,
           0.04520568,  0.15955793,  0.14916318,  0.07530025,  0.01073891,
           -0.01989327,  0.01197282,  0.01746466,  0.01909844, -0.02407202,
           0.02229063, -0.02624946, -0.02769674,  0.02380409,  0.01521666,
           0.00523155,  0.01721179,  0.01374679,  0.03257269, -0.03502101,
           0.05376244, -0.01104296,  0.01409695,  0.00738489],

          [0.06469896,  0.07474376, -0.10497157, -0.01163837,  0.10664633,
          0.08122487,  0.14961286,  0.01129187,  0.03269974,  0.00542278,
          0.02990973, -0.01245547, -0.02566625,  0.02235499,  0.00775378,
          0.04040706,  0.01026787,  0.03104725,  0.07073719,  0.09502772,
          0.06318198,  0.05780239,  0.00831813,  0.00795687,  0.00152939,
          0.07464983,  0.04592341,  0.06990428,  0.03093214],

          [0.05257595,  0.03645374,  0.05064316,  0.09915976,  0.12516487,
          0.04200103,  0.0507318,  0.05682209,  0.00084628, -0.00966819,
          0.02206698,  0.01726283,  0.0389894,  0.0232979,  0.04128902,
          0.11674435,  0.03812969,  0.03572417,  0.02063299, -0.04311839,
          0.01866072,  0.04061572,  0.03144334,  0.02816955,  0.0047644,
          0.03373313,  0.01036463,  0.01330643,  0.01620241],

          [0.01502225,  0.02574921,  0.04296581,  0.01661686,  0.11951815,
           0.06505257,  0.05568408,  0.01874875, -0.03594564,  0.02925688,
           0.00554119,  0.02740969,  0.01539766,  0.02117471, -0.00487856,
           0.03082656,  0.02684759,  0.01661798,  0.05359333,  0.12494067,
           0.07820925,  0.01797435, -0.00658785, -0.00048644,  0.04288145,
           0.05319086,  0.04419764,  0.06383189,  0.06701227],

          [7.79929026e-02,  5.67386991e-02,  1.44929608e-01,  1.34492005e-01,
          1.12750180e-01,  3.56229446e-02,  1.41477858e-01,  5.95891275e-02,
          -2.50007118e-02,  3.93585935e-02,  3.49288585e-02, -1.70993974e-02,
          -5.27157498e-03,  2.10764370e-02,  1.38856889e-02,  5.17411797e-02,
          4.67630576e-02,  6.14151896e-02, -2.25034385e-02, -2.25179271e-02,
          -3.69107059e-02, -1.12809649e-02,  1.50095758e-02, -1.22704096e-04,
          3.65159796e-02,  2.07487457e-02,  7.87955764e-03,  4.13478776e-02,
          2.03933043e-02],

          [0.04338571,  0.04725584,  0.09743032,  0.09762056,  0.09758751,
          0.10276827,  0.06671437,  0.03458499,  0.02820957,  0.03649327,
          0.01013807,  0.00729155,  0.01008569,  0.03306697,  0.01687073,
          0.02373477,  0.03305804,  0.03296769, -0.01751235,  0.0154248,
          0.00192107,  0.02958529,  0.01304771,  0.01283894,  0.0259287,
          0.02284008,  0.02568435,  0.0328813,  0.05444213],

          [0.05419176, -0.03096007,  0.07400283,  0.07455961,  0.17137121,
          0.10105855,  0.11509882,  0.01648057,  0.03468728,  0.08206215,
          0.03059283,  0.02908416,  0.04209344,  0.05760641, -0.00220013,
          0.03623844,  0.06615092,  0.07719467, -0.01944284,  0.00559424,
          -0.02400303,  0.00390466, -0.00976022, -0.0142846, -0.00365805,
          0.00251611, -0.00072374,  0.01152074,  0.02241151]

          ]

preds_test = np.stack(preds_test, axis=2)
df_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
preds_test_ensemble = preds_test[:, :, 0].copy()
for n, w in enumerate(w_list):
    if n == 6:
        preds_test_ensemble[:, n * 60:] = (preds_test[:, n * 60:, :]
                                           * np.array(w).reshape(1, 1, -1)).sum(axis=2)
    else:
        preds_test_ensemble[:, n * 60: (n + 1) * 60] = (preds_test[:, n * 60: (
            n + 1) * 60, :] * np.array(w).reshape(1, 1, -1)).sum(axis=2)
df_sub[TARGET_COLUMNS] = preds_test_ensemble
df_sub.to_parquet(exp_dir / f"20240715_ensemble_per_target_{max_iter}.parquet")
