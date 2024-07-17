# ==================
# Libraries
# ==================
from pathlib import Path
import os
import random
from transformers import AdamW, get_cosine_schedule_with_warmup
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
exp = "124"
exp_dir = OUTPUT_DIR / "exp" / f"ex{exp}"
model_dir = exp_dir / "model"

exp_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
logger_path = exp_dir / f"ex{exp}.txt"

# config
seed = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model config
batch_size = 3
n_epochs = 20
lr = 1e-3
weight_decay = 0.05
num_warmup_steps = 1000

# =====================
# Data
# =====================
data_fe = "126"
data_path = OUTPUT_DIR / "fe" / \
    f"fe{data_fe}" / f"fe{data_fe}_data_list.parquet"

data_fe2 = "127"
data_path2 = OUTPUT_DIR / "fe" / \
    f"fe{data_fe2}" / f"fe{data_fe2}_data_list.parquet"

data_fe3 = "128"
data_path3 = OUTPUT_DIR / "fe" / \
    f"fe{data_fe3}" / f"fe{data_fe3}_data_list.parquet"

data_fe4 = "145"
data_path4 = OUTPUT_DIR / "fe" / \
    f"fe{data_fe4}" / f"fe{data_fe4}_data_list.parquet"
# =====================
# Funcsion
# =====================


class LeapDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, item):
        prefix = self.data_path[item]
        seq_array = np.load(prefix[3:] + "_seq_feature.npy")
        other_array = np.load(prefix[3:] + "_other_feature.npy")
        target_seq_array = np.load(prefix[3:] + "_seq_target.npy")
        other_target_array = np.load(prefix[3:] + "_other_target.npy")

        return {
            'input_data_seq_array': torch.tensor(
                seq_array, dtype=torch.float32),
            'input_data_other_array': torch.tensor(
                other_array, dtype=torch.float32),
            'target_seq_array': torch.tensor(
                target_seq_array, dtype=torch.float32),
            'target_other_array': torch.tensor(
                other_target_array, dtype=torch.float32)}


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

data_list1 = pd.read_parquet(data_path)
data_list2 = pd.read_parquet(data_path2)
data_list3 = pd.read_parquet(data_path3)
data_list4 = pd.read_parquet(data_path4)
data_list = pd.concat([data_list1, data_list2, data_list3, data_list4],
                      axis=0).reset_index(drop=True)

# 学習
with timer("rnn"):
    set_seed(seed)

    train_ = LeapDataset(data_list["data_path"].values)

    train_loader = DataLoader(dataset=train_,
                              batch_size=batch_size,
                              shuffle=True)

    model = LeapRnnModel()
    model = model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)],
            'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=lr,
                      weight_decay=weight_decay,
                      )
    num_train_optimization_steps = int(len(train_loader) * n_epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_optimization_steps)
    criterion = nn.SmoothL1Loss()
    best_val_score = 0
    for epoch in range(n_epochs):
        model.train()
        train_losses_batch = []
        val_losses_batch = []
        epoch_loss = 0
        for d in tqdm(train_loader, total=len(train_loader)):
            # =========================
            # data loader
            # =========================
            input_data_seq_array = d[
                'input_data_seq_array'].to(
                device)
            input_data_other_array = d[
                'input_data_other_array'].to(
                device)
            target_seq_array = d["target_seq_array"].to(device)
            target_other_array = d["target_other_array"].to(device)
            input_data_seq_array = input_data_seq_array.view(
                -1, 60, 18)
            input_data_other_array = input_data_other_array.view(
                -1, 17)
            target_seq_array = target_seq_array.view(-1, 60, 6)
            target_other_array = target_other_array.view(-1, 8)
            optimizer.zero_grad()

            output_seq, output_other = model(input_data_seq_array,
                                             input_data_other_array)
            output_seq = output_seq.view(-1, 60 * 6)
            output = torch.cat([output_seq, output_other], dim=1)
            target_seq_array = target_seq_array.view(-1, 60 * 6)
            target = torch.cat([target_seq_array, target_other_array], dim=1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses_batch.append(loss.item())
        train_loss = np.mean(train_losses_batch)

        # ==========================
        # eval
        # ==========================

        LOGGER.info(
            f"epoch:{epoch},train loss:{train_loss}")
        torch.save(model.state_dict(), model_dir /
                   f"ex{exp}_{epoch}.pth")
