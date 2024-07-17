# ==================
# Libraries
# ==================
import pickle
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
import pandas as pd

# =========================
# Constants
# =========================
OUTPUT_DIR = Path("storage/leap/output")
DATA_DIR = Path("storage/leap/data")

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
exp = "138"
exp_dir = OUTPUT_DIR / "exp" / f"ex{exp}"
model_dir = exp_dir / "model"

exp_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
logger_path = exp_dir / f"ex{exp}.txt"

TEST_PATH1 = OUTPUT_DIR / "fe" / "fe100" / "fe100_test.parquet"

# config
batch_size = 1024
seed = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# Data
# =====================
data_fe = "134"
data_dir = OUTPUT_DIR / "fe" / f"fe{data_fe}"

data_diff_fe = "135"
data_diff_dir = OUTPUT_DIR / "fe" / f"fe{data_diff_fe}"

data_add_fe = "136"
data_add_dir = OUTPUT_DIR / "fe" / f"fe{data_add_fe}"

# sc_dict
train_target_path = OUTPUT_DIR / "fe" / "fe101" / "fe101_target_mean_std.pkl"

# feature
seq_list = [
    data_dir / f"fe{data_fe}_state_t.npy",
    data_dir / f"fe{data_fe}_state_q0001.npy",
    data_dir / f"fe{data_fe}_state_q0002.npy",
    data_dir / f"fe{data_fe}_state_q0003.npy",
    data_dir / f"fe{data_fe}_state_u.npy",
    data_dir / f"fe{data_fe}_state_v.npy",
    data_dir / f"fe{data_fe}_pbuf_ozone.npy",
    data_dir / f"fe{data_fe}_pbuf_CH4.npy",
    data_dir / f"fe{data_fe}_pbuf_N2O.npy"
]

seq_diff_list = [
    data_diff_dir / f"fe{data_diff_fe}_state_t_diff.npy",
    data_diff_dir / f"fe{data_diff_fe}_state_q0001_diff.npy",
    data_diff_dir / f"fe{data_diff_fe}_state_q0002_diff.npy",
    data_diff_dir / f"fe{data_diff_fe}_state_q0003_diff.npy",
    data_diff_dir / f"fe{data_diff_fe}_state_u_diff.npy",
    data_diff_dir / f"fe{data_diff_fe}_state_v_diff.npy",
    data_diff_dir / f"fe{data_diff_fe}_pbuf_ozone_diff.npy",
    data_diff_dir / f"fe{data_diff_fe}_pbuf_CH4_diff.npy",
    data_diff_dir / f"fe{data_diff_fe}_pbuf_N2O_diff.npy"
]

seq_list += seq_diff_list

other_path = data_dir / f"fe{data_fe}_other.npy"
add_path = data_add_dir / f"fe{data_add_fe}_state_sum.npy"

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
                               model_size * 2, kernel_size=9, padding=4)
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


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s')


setup_logger(out_file=logger_path)

# =====================
# Main
# =====================

# seq
seq_feature = []
for p in tqdm(seq_list):
    tmp = np.load(p)
    tmp = tmp.reshape([-1, 60, 1])
    seq_feature.append(tmp)

seq_feature = np.concatenate(seq_feature, axis=2)
other_feature = np.load(other_path)
add_feature = np.load(add_path)
other_feature = np.concatenate(
    [other_feature, add_feature.reshape(-1, 1)], axis=1)
# target
with open(train_target_path, 'rb') as f:
    target_sc_dict = pickle.load(f)

val_seq_feature = seq_feature.copy()
val_other_feature = other_feature.copy()


with timer("inference"):
    set_seed(seed)

    val_ = LeapDataset(val_seq_feature,
                       val_other_feature)

    val_loader = DataLoader(dataset=val_,
                            batch_size=batch_size * 2,
                            shuffle=False, num_workers=8)

    model = LeapRnnModel()
    model.load_state_dict(torch.load(model_dir /
                                     f"ex{exp}.pth"))
    model.to(device)
    colapse_col = np.load(exp_dir / f"exp{exp}_clapse_col.npy")

    # ==========================
    # eval
    # ==========================

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
    for n, c in enumerate(target_cols):
        for i in range(60):
            seq_val_preds[:, i, n] = seq_val_preds[:, i, n] * \
                target_sc_dict[f"{c}_{i}"][1] + \
                target_sc_dict[f"{c}_{i}"][0]
    for n, c in enumerate(other_target_cols):
        other_val_preds[:, n] = other_val_preds[:, n] * \
            target_sc_dict[c][1] + target_sc_dict[c][0]

    seq_val_preds[:, :12, 1] = 0
    seq_val_preds[:, :27, 2] = 0
    seq_val_preds[:, :12, 3] = 0
    seq_val_preds[:, :12, 4] = 0
    seq_val_preds[:, :12, 5] = 0
    seq_val_pred_flat = np.zeros([len(seq_val_preds), 6 * 60])
    for i in range(6):
        seq_val_pred_flat[:, i * 60: (i + 1) * 60] = seq_val_preds[:, :, i]

    val_preds_all = np.concatenate(
        [seq_val_pred_flat, other_val_preds], axis=1)
    r2_all = []
    for i in colapse_col:
        val_preds_all[:, i] = 0

test_pred = pd.DataFrame(val_preds_all)
ptend_t = [f'ptend_t_{i}' for i in range(60)]
ptend_q0001 = [f'ptend_q0001_{i}' for i in range(60)]
ptend_q0002 = [f'ptend_q0002_{i}' for i in range(60)]
ptend_q0003 = [f'ptend_q0003_{i}' for i in range(60)]
ptend_u = [f'ptend_u_{i}' for i in range(60)]
ptend_v = [f'ptend_v_{i}' for i in range(60)]
other_target = ['cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC',
                'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL',
                'cam_out_SOLSD', 'cam_out_SOLLD']
target_list = [ptend_t, ptend_q0001, ptend_q0002,
               ptend_q0003, ptend_u, ptend_v, other_target]
target_cols = []
for c in target_list:
    for i in c:
        target_cols.append(i)
test_pred.columns = target_cols

sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
test_pred["sample_id"] = sub["sample_id"]
test_pred = test_pred[sub.columns]

for c in tqdm(test_pred.columns):
    if c != "sample_id":
        test_pred[c] = test_pred[c].astype(np.float32)

REPLACE_FROM = ['ptend_q0002_0', 'ptend_q0002_1', 'ptend_q0002_2', 'ptend_q0002_3',
                'ptend_q0002_4', 'ptend_q0002_5', 'ptend_q0002_6', 'ptend_q0002_7',
                'ptend_q0002_8',
                'ptend_q0002_9', 'ptend_q0002_10',
                'ptend_q0002_11', 'ptend_q0002_12', 'ptend_q0002_13',
                'ptend_q0002_14', 'ptend_q0002_15',
                'ptend_q0002_16', 'ptend_q0002_17', 'ptend_q0002_18',
                'ptend_q0002_19', 'ptend_q0002_20',
                'ptend_q0002_21', 'ptend_q0002_22', 'ptend_q0002_23', 'ptend_q0002_24',
                'ptend_q0002_25', 'ptend_q0002_26', 'ptend_q0002_27']
REPLACE_TO = ['state_q0002_0', 'state_q0002_1', 'state_q0002_2', 'state_q0002_3',
              'state_q0002_4', 'state_q0002_5',
              'state_q0002_6', 'state_q0002_7', 'state_q0002_8', 'state_q0002_9',
              'state_q0002_10', 'state_q0002_11', 'state_q0002_12',
              'state_q0002_13', 'state_q0002_14', 'state_q0002_15', 'state_q0002_16',
              'state_q0002_17', 'state_q0002_18', 'state_q0002_19',
              'state_q0002_20', 'state_q0002_21', 'state_q0002_22', 'state_q0002_23',
              'state_q0002_24', 'state_q0002_25', 'state_q0002_26', 'state_q0002_27']
test = pd.read_parquet(TEST_PATH1)
static_pred = -test[REPLACE_TO].values * sub[REPLACE_FROM].values / 1200
for c in REPLACE_FROM:
    test_pred[c] = test_pred[c].astype(np.float64)
test_pred[REPLACE_FROM] = static_pred
test_pred.to_parquet(exp_dir / f"ex{exp}_pp.parquet")
