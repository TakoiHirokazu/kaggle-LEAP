# ==================
# Libraries
# ==================
from pathlib import Path
import os
import random
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import logging
from contextlib import contextmanager
import time
import sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score
import gc
import pickle


# =========================
# Constants
# =========================
DATA_DIR = Path("storage/leap/data/")
OUTPUT_DIR = Path("storage/leap/output")

ptend_t = [f'ptend_t_{i}' for i in range(60)]
ptend_q0001 = [f'ptend_q0001_{i}' for i in range(60)]
ptend_q0002 = [f'ptend_q0002_{i}' for i in range(60)]
ptend_q0003 = [f'ptend_q0003_{i}' for i in range(60)]
ptend_u = [f'ptend_u_{i}' for i in range(60)]
ptend_v = [f'ptend_v_{i}' for i in range(60)]

target_cols = ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003',
               'ptend_u', 'ptend_v']
other_target_cols = ['cam_out_NETSW', 'cam_out_FLWDS',
                     'cam_out_PRECSC', 'cam_out_PRECC', 'cam_out_SOLS',
                     'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']

# =========================
# Settings
# =========================
exp = "134"
exp_dir = OUTPUT_DIR / "exp" / f"ex{exp}"
model_dir = exp_dir / "model"

exp_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
logger_path = exp_dir / f"ex{exp}.txt"

# config
seed = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model config
batch_size = 1024
n_epochs = 20
lr = 1e-3
weight_decay = 0.05
num_warmup_steps = 1000

# =====================
# Data
# =====================

data_fe2 = "130"
data_dir2 = OUTPUT_DIR / "fe" / f"fe{data_fe2}"

data_diff_fe2 = "131"
data_diff_dir2 = OUTPUT_DIR / "fe" / f"fe{data_diff_fe2}"

data_add_fe2 = "132"
data_add_dir2 = OUTPUT_DIR / "fe" / f"fe{data_add_fe2}"

# sc_dict
train_target_path = OUTPUT_DIR / "fe" / "fe101" / "fe101_target_mean_std.pkl"


# feature

seq_list2 = [
    data_dir2 / f"fe{data_fe2}_state_t.npy",
    data_dir2 / f"fe{data_fe2}_state_q0001.npy",
    data_dir2 / f"fe{data_fe2}_state_q0002.npy",
    data_dir2 / f"fe{data_fe2}_state_q0003.npy",
    data_dir2 / f"fe{data_fe2}_state_u.npy",
    data_dir2 / f"fe{data_fe2}_state_v.npy",
    data_dir2 / f"fe{data_fe2}_pbuf_ozone.npy",
    data_dir2 / f"fe{data_fe2}_pbuf_CH4.npy",
    data_dir2 / f"fe{data_fe2}_pbuf_N2O.npy"
]


seq_diff_list2 = [
    data_diff_dir2 / f"fe{data_diff_fe2}_state_t_diff.npy",
    data_diff_dir2 / f"fe{data_diff_fe2}_state_q0001_diff.npy",
    data_diff_dir2 / f"fe{data_diff_fe2}_state_q0002_diff.npy",
    data_diff_dir2 / f"fe{data_diff_fe2}_state_q0003_diff.npy",
    data_diff_dir2 / f"fe{data_diff_fe2}_state_u_diff.npy",
    data_diff_dir2 / f"fe{data_diff_fe2}_state_v_diff.npy",
    data_diff_dir2 / f"fe{data_diff_fe2}_pbuf_ozone_diff.npy",
    data_diff_dir2 / f"fe{data_diff_fe2}_pbuf_CH4_diff.npy",
    data_diff_dir2 / f"fe{data_diff_fe2}_pbuf_N2O_diff.npy"
]


seq_list2 += seq_diff_list2

state_q002_raw_list = [data_dir2 / f"fe{data_fe2}_state_q0002_raw.npy"]

other_path2 = data_dir2 / f"fe{data_fe2}_other.npy"
add_path2 = data_add_dir2 / f"fe{data_add_fe2}_state_sum.npy"

# target
target_list2 = [
    data_dir2 / f"fe{data_fe2}_ptend_t_target.npy",
    data_dir2 / f"fe{data_fe2}_ptend_q0001_target.npy",
    data_dir2 / f"fe{data_fe2}_ptend_q0002_target.npy",
    data_dir2 / f"fe{data_fe2}_ptend_q0003_target.npy",
    data_dir2 / f"fe{data_fe2}_ptend_u_target.npy",
    data_dir2 / f"fe{data_fe2}_ptend_v_target.npy",
]

other_target_path2 = data_dir2 / f"fe{data_fe2}_other_target.npy"


target_raw_list2 = [
    data_dir2 / f"fe{data_fe2}_ptend_t_target_raw.npy",
    data_dir2 / f"fe{data_fe2}_ptend_q0001_target_raw.npy",
    data_dir2 / f"fe{data_fe2}_ptend_q0002_target_raw.npy",
    data_dir2 / f"fe{data_fe2}_ptend_q0003_target_raw.npy",
    data_dir2 / f"fe{data_fe2}_ptend_u_target_raw.npy",
    data_dir2 / f"fe{data_fe2}_ptend_v_target_raw.npy",
]
other_target_raw_path2 = data_dir2 / f"fe{data_fe2}_other_target_raw.npy"
# =====================
# Funcsion
# =====================


class LeapDataset(Dataset):
    def __init__(self, seq_array,
                 other_array):
        self.seq_array = seq_array
        self.other_array = other_array

    def __len__(self):
        return len(self.seq_array)

    def __getitem__(self, item):

        return {
            'input_data_seq_array': torch.tensor(
                self.seq_array[item], dtype=torch.float32),
            'input_data_other_array': torch.tensor(
                self.other_array[item], dtype=torch.float32)
        }


class LeapRnnModel(nn.Module):
    def __init__(
            self,
            input_numerical_size=9 * 2,
            numeraical_linear_size=64,
            input_numerical_size2=17,
            numeraical_linear_size2=64,
            model_size=256 * 2,
            linear_out=256,
            out_size1=6,
            out_size2=8):
        super(LeapRnnModel, self).__init__()
        self.numerical_linear = nn.Sequential(
            nn.Linear(input_numerical_size,
                      numeraical_linear_size),
            nn.LayerNorm(numeraical_linear_size)
        )
        self.numerical_linear2_list = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(input_numerical_size2,
                          numeraical_linear_size2),
                nn.LayerNorm(numeraical_linear_size2)
            ) for _ in range(60)]
        )
        self.numerical_linear2 = nn.Sequential(
            nn.Linear(input_numerical_size2,
                      numeraical_linear_size2),
            nn.LayerNorm(numeraical_linear_size2)
        )
        self.rnn = nn.LSTM(numeraical_linear_size + numeraical_linear_size2,
                           model_size,
                           num_layers=3,
                           batch_first=True,
                           bidirectional=True)
        self.conv1 = nn.Conv1d(model_size * 2,
                               model_size * 2, kernel_size=5, padding=2)
        self.linear_out1 = nn.Sequential(
            nn.Linear(model_size * 2,
                      linear_out),
            nn.LayerNorm(linear_out),
            nn.ReLU(),
            nn.Linear(linear_out,
                      out_size1))
        self.layernorm = nn.LayerNorm(model_size * 2)
        self.linear_out2 = nn.Sequential(
            nn.Linear(model_size * 2 + numeraical_linear_size2,
                      linear_out),
            nn.LayerNorm(linear_out),
            nn.ReLU(),
            nn.Linear(linear_out,
                      out_size2))
        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'rnn' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)

    def forward(self, seq_array,
                other_array):

        numerical_embedding = self.numerical_linear(seq_array)
        other_embedding = self.numerical_linear2(other_array)
        numerical_embedding2_list = [
            linear(other_array) for linear in self.numerical_linear2_list]
        numerical_embedding2 = torch.stack(numerical_embedding2_list, dim=1)
        numerical_embedding_concat = torch.cat(
            [numerical_embedding, numerical_embedding2], dim=2)
        output_seq, _ = self.rnn(numerical_embedding_concat)
        output_seq = output_seq.permute(
            (0, 2, 1)).contiguous()
        output_seq = self.conv1(output_seq)
        output_seq = output_seq.permute(
            (0, 2, 1)).contiguous()
        output_other = torch.mean(output_seq, dim=1)
        output_other = self.layernorm(output_other)
        output_other = torch.cat([output_other, other_embedding], dim=1)
        output_seq = self.linear_out1(output_seq)
        output_other = self.linear_out2(output_other)
        return output_seq, output_other


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


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


@ contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s')


setup_logger(out_file=logger_path)


# =====================
# Main
# =====================


seq_feature2 = []
for p in tqdm(seq_list2):
    tmp = np.load(p)
    tmp = tmp.reshape([-1, 60, 1])
    seq_feature2.append(tmp)
seq_feature2 = np.concatenate(seq_feature2, axis=2)
other_feature2 = np.load(other_path2)
add_feature2 = np.load(add_path2)

seq_feature = seq_feature2.copy()
other_feature = other_feature2.copy()
add_feature = add_feature2.copy()
other_feature = np.concatenate(
    [other_feature, add_feature.reshape(-1, 1)], axis=1)
del seq_feature2, other_feature2, add_feature2

state_q002_raw = []
for p in state_q002_raw_list:
    tmp = np.load(p)
    state_q002_raw.append(tmp)
state_q002_raw = np.concatenate(state_q002_raw, axis=0)


# train val split

val_seq_feature = seq_feature.copy()
val_other_feature = other_feature.copy()
val_state_q002_raw = state_q002_raw.copy()


target_raw_seq2 = []
for p in tqdm(target_raw_list2):
    tmp = np.load(p)
    tmp = tmp.reshape([-1, 60, 1])
    target_raw_seq2.append(tmp)
target_raw_seq2 = np.concatenate(target_raw_seq2, axis=2)
other_raw_target2 = np.load(other_target_raw_path2)

target_raw_seq = target_raw_seq2.copy()
other_raw_target = other_raw_target2.copy()
del target_raw_seq2, other_raw_target2

val_target_raw_seq = target_raw_seq.copy()
val_other_raw_target = other_raw_target.copy()
del target_raw_seq, other_raw_target
gc.collect()
# 最終評価ように変換
val_target_seq_ = val_target_raw_seq.copy()
val_other_target_ = val_other_raw_target.copy()
val_target_flat = np.zeros([len(val_target_seq_), 6 * 60])
for i in range(6):
    val_target_flat[:, i * 60: (i + 1) * 60] = val_target_seq_[:, :, i]
val_target_all = np.concatenate([val_target_flat, val_other_target_], axis=1)
del val_target_flat, val_other_target_
gc.collect()


with open(train_target_path, 'rb') as f:
    target_sc_dict = pickle.load(f)


with timer("lstm"):
    set_seed(seed)

    val_ = LeapDataset(val_seq_feature,
                       val_other_feature)
    val_loader = DataLoader(dataset=val_,
                            batch_size=batch_size * 2,
                            shuffle=False, num_workers=8)

    best_val_score = 0
    for epoch in range(17, 20):
        model = LeapRnnModel()
        model.load_state_dict(torch.load(model_dir /
                                         f"ex{exp}_{epoch}.pth"))
        model = model.to(device)
        val_losses_batch = []
        model.eval()  # switch model to the evaluation mode
        seq_val_preds = []
        other_val_preds = []
        with torch.no_grad():
            # Predicting on validation set
            for d in val_loader:
                input_data_seq_array = d[
                    'input_data_seq_array'].to(
                    device)
                input_data_other_array = d[
                    'input_data_other_array'].to(
                    device)
                output_seq, output_other = model(input_data_seq_array,
                                                 input_data_other_array)
                seq_val_preds.append(
                    output_seq.detach().cpu().numpy().astype(np.float64))
                other_val_preds.append(
                    output_other.detach().cpu().numpy().astype(np.float64))
        seq_val_preds = np.concatenate(seq_val_preds, axis=0)
        other_val_preds = np.concatenate(other_val_preds, axis=0)
        seq_val_preds_ = seq_val_preds.copy()
        other_val_preds_ = other_val_preds.copy()
        for n, c in enumerate(target_cols):
            for i in range(60):
                seq_val_preds[:, i, n] = seq_val_preds[:, i, n] * \
                    target_sc_dict[f"{c}_{i}"][1] + \
                    target_sc_dict[f"{c}_{i}"][0]
        for n, c in enumerate(other_target_cols):
            other_val_preds[:, n] = other_val_preds[:, n] * \
                target_sc_dict[c][1] + target_sc_dict[c][0]

        seq_val_preds[:, :12, 1] = 0
        seq_val_preds[:, :12, 2] = 0 * \
            val_state_q002_raw[:, :12] / (-1200)
        seq_val_preds[:, 12:26, 2] = 1.0 * \
            val_state_q002_raw[:, 12:26] / (-1200)
        seq_val_preds[:, 26, 2] = 1.0 * \
            val_state_q002_raw[:, 26] / (-1200)
        seq_val_preds[:, 27, 2] = 1.0 * \
            val_state_q002_raw[:, 27] / (-1200)
        seq_val_preds[:, :12, 3] = 0
        seq_val_preds[:, :12, 4] = 0
        seq_val_preds[:, :12, 5] = 0
        seq_val_pred_flat = np.zeros([len(seq_val_preds), 6 * 60])
        for i in range(6):
            seq_val_pred_flat[:, i * 60: (i + 1) * 60] = seq_val_preds[:, :, i]

        val_preds_all = np.concatenate(
            [seq_val_pred_flat, other_val_preds], axis=1)
        r2_all = []
        clapse_col = []
        for i in range(val_target_all.shape[1]):
            cv = r2_score(val_target_all[:, i], val_preds_all[:, i])
            if cv < 0:
                val_preds_all[:, i] = 0
                clapse_col.append(i)
            cv = r2_score(val_target_all[:, i], val_preds_all[:, i])
            r2_all.append(cv)
        cv = np.mean(r2_all)
        all_cv = r2_score(val_target_all, val_preds_all)
        r2_list = []
        for n, c in enumerate(target_cols):
            pred_ = val_preds_all[:, n * 60: (n + 1) * 60]
            target_ = val_target_all[:, n * 60:(n + 1) * 60]
            r2_ = r2_score(target_, pred_)
            r2_list.append(r2_)
        pred_ = val_preds_all[:, 360:]
        target_ = val_target_all[:, 360:]
        r2_ = r2_score(target_, pred_)
        r2_list.append(r2_)
        LOGGER.info(
            f"epoch:{epoch},cv:{cv}:{all_cv}")
        LOGGER.info(f"r2_list:{r2_list}")
        if cv >= best_val_score:
            LOGGER.info("save weight")
            best_val_score = cv
            best_val_preds = val_preds_all.copy()
            best_seq_val_preds = seq_val_preds_.copy()
            best_other_val_preds = other_val_preds_.copy()
            clapse_col_ = clapse_col.copy()
            torch.save(model.state_dict(), model_dir /
                       f"ex{exp}.pth")

    np.save(exp_dir / f"exp{exp}_val_preds.npy", best_val_preds)
    np.save(exp_dir / f"exp{exp}_seq_pred.npy", best_seq_val_preds)
    np.save(exp_dir / f"exp{exp}_other_pred.npy", best_other_val_preds)
    np.save(exp_dir / f"exp{exp}_clapse_col.npy", np.array(clapse_col_))
