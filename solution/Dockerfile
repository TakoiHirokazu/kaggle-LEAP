FROM nvcr.io/nvidia/pytorch:23.11-py3

# ライブラリの追加インストール
RUN pip install -U pip && \
    pip install iterative-stratification==0.1.7 torch-ema==0.3 transformers==4.36.2 lightgbm==4.2.0 \
    catboost==1.2.2 optuna==3.5.0 polars==0.20.18 gensim==4.3.2 sentencepiece==0.1.99 holidays==0.40 \
    pyarrow==16.1.0
    
WORKDIR /tmp/working
