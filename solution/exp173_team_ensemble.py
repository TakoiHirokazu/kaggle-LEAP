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
    pred_val = pd.read_parquet(
        f"{folder}/pred_valid.parquet")[TARGET_COLUMNS].values[-n_data_for_eval:].astype(np.float64)
    pred_test = pd.read_parquet(
        f"{folder}/submission.parquet")[TARGET_COLUMNS].values.astype(np.float64)

    # print(folder, pred_val.shape, pred_test.shape, pred_val.dtype, pred_test.dtype)

    preds_val.append(pred_val)
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
    pred_val = np.load(file)[-n_data_for_eval:]
    pred_val = pred_val / weight.reshape(1, -1)
    pred_val[np.isnan(pred_val)] = 0

    pred_test = pd.read_parquet(takoi_files_test[n])[TARGET_COLUMNS].values

    print(file, pred_val.shape, pred_test.shape,
          pred_val.dtype, pred_test.dtype)
    preds_val.append(pred_val)
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
    pred_val = pd.read_parquet(file)[TARGET_COLUMNS].values[-n_data_for_eval:]
    pred_val = pred_val / weight.reshape(1, -1)
    pred_val[np.isnan(pred_val)] = 0

    pred_test = pd.read_parquet(file.replace(
        "_valid_pred.parquet", "_submission.parquet"))[TARGET_COLUMNS].values

    print(file, pred_val.shape, pred_test.shape,
          pred_val.dtype, pred_test.dtype)
    preds_val.append(pred_val)
    preds_test.append(pred_test)
    exp_names.append(file)

preds_val = np.stack(preds_val, axis=2)


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


ALL_ZERO_INDICE = [
    np.where(TARGET_COLUMNS == col)[0] for col in ALL_ZERO_TARGET_COLS
]

preds_val[:, ALL_ZERO_INDICE] = 0
label[:, ALL_ZERO_INDICE] = 0

rets = []
models = []
for i in tqdm.tqdm(range(preds_val.shape[2])):
    ret = {
        "exp_name": exp_names[i],
        "r2": r2_score(label, preds_val[:, :, i])
    }
    rets.append(ret)


n_models = preds_val.shape[2]
kurupical_weights = np.ones(len(kurupical_folders)) / len(kurupical_folders)
takoi_weights = np.ones(len(takoi_files)) / len(takoi_files)
kami_weights = np.ones(len(kami_files)) / len(kami_files)

weights = np.concatenate([kurupical_weights, takoi_weights, kami_weights])
weights /= weights.sum()

rets.append({
    "exp_name": "average",
    "r2": r2_score(label, (preds_val * weights.reshape(1, 1, -1)).sum(axis=2))
})

pred_val_ = preds_val.astype(np.float32)
label_ = label.astype(np.float32)
w_list = []
for i in range(7):
    with timer(f"start {i}"):
        if i == 6:
            ret = nelder_mead_optimization(
                pred_val_[:, 60 * i:, :], label_[:, 60 * i:], max_iterations=max_iter)
        else:
            ret = nelder_mead_optimization(pred_val_[
                                           :, 60 * i: 60 * (i + 1), :], label_[:, 60 * i: 60 * (i + 1)], max_iterations=max_iter)
        # ret = nelder_mead_optimization(pred_val_, label_, max_iterations=5)
        w_list.append(list(ret["weights"]))
        LOGGER.info(
            f"{i},{ret}")
