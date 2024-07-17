# Kaggle Solution Summary: takoi Part

## Summary
- Used data from Hugging Face's LEAP/ClimSim_low-res for training
- Created features from the data in a time series format
- Developed 12 models based on LSTM

## Features Selection / Engineering
- Data
    - Validation: Used the last 639,744 rows from Kaggle's train.csv
    - Train: Used Hugging Face's LEAP/ClimSim_low-res data (excluding the 9th year and periods overlapping with the validation period)

- Features
    - Divided into time series parts and others
    - Time series parts (considered as a series of length 60)
        - Original data
        - Differences from the subsequent data points in the series
    - Other parts
        - Original data
        - Sum of state_q0001, state_q0002, and state_q0003

- Preprocessing
    - Features
        - StandardScaler
            - Applied StandardScaler to each column
    - Target
        - Used the value after applying weight to the target
        - StandardScaler
            - Applied StandardScaler to each column

## models
- The models are based on LSTM
- The following is the base model
    - Ultimately, multiple models were created by enlarging the base model or adding conv1d
```
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

```

### Training Method
- loss : SmoothL1Loss
- scheduler : get_cosine_schedule_with_warmup
- optimizer : AdamW
- lr : 1e-3

### Post-Processing
- Replace values of ptend_q0002_0 to ptend_q0002_27 with the corresponding values of state_q0002_0 to state_q0002_27 divided by (-1200).
- Set columns with a weight of 0 in sample_submission.csv to 0.

### Model Results
- CV is evaluated using the validation data
    - Results after post-processing
- Some experiments do not use the entire dataset mentioned above, so the approximate size of the training data is noted

| # | exp no | CV | Training Time(h) | Data Size | Note |
| --- | --- | --- | --- | --- |--- |
| 1 | 124| 0.7812|  24h | about 45M ||
| 2 | 130 |0.7815 |  30h | about 45M |add 1dcnn|
| 3 | 131 |0.7816  | 34h | about 60M ||
| 4 | 133 |0.7819  | 40h | about 60M| add 1dcnn|
| 5 | 134 |0.7817  | 36h | about 60M  |add 1dcnn|
| 6 | 135 |0.7819  | 42h | about 70M ||
| 7 | 136 |0.7821  | 45h | about 70M |add 1dcnn |
| 8 | 138 |0.7824  | 51h | about 70M |add 1dcnn|
| 9 | 139 | 0.7822 | 59h | about 70M |add 1dcnn |
| 10 | 141 |0.7825  | 116h | about 70M |add 1dcnn + large model|
| 11 | 159 |0.7827  | 52h | about 70M |add 1dcnn|
| 1２ | 162 |0.7838  | 47h |about 70M  |add 1dcnn + large model|

### Ensemble
- The ensemble weights are determined using Nelder-Mead based on the predictions from team members.
- The weights are optimized for each target group (such as ptend_t or ptend_q0001).
- The ensemble results are used to replace the predictions from team member Kurupical's stacking.



