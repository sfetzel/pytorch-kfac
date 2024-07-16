#!/bin/bash

PLANETOID_DATASETS="Cora CiteSeer PubMed"

# introduction experiments for GCNs:
KIPF_RUNS=100 # 100 in paper.
for dataset in $PLANETOID_DATASETS; do
  python planetoid.py --cov_update_freq=1 --model=GCN --dropout=0.5 --weight_decay=5e-4\
    --hidden_channels=16 --kfac_damping=0.1 --lr=0.01 --cov_ema_decay=0.25 \
    --baseline=adam --dataset=$dataset --runs=$KIPF_RUNS --file_name=results/GCN-$dataset-kipf
done
VELICKOVIC_RUNS=100 #100
# introduction experiments for GATs:
for dataset in Cora CiteSeer; do
  python planetoid.py --cov_update_freq=1 --model=GAT --dropout=0.6 --weight_decay=5e-4\
    --hidden_channels=8 --heads=8 --kfac_damping=0.1 --lr=0.005 --cov_ema_decay=0.25 \
    --baseline=adam --dataset=$dataset --runs=$VELICKOVIC_RUNS --file_name=results/GAT-$dataset-Veličković
done

dataset=PubMed
python planetoid.py --cov_update_freq=1 --model=GAT --dropout=0.6 --weight_decay=0.001\
    --hidden_channels=8 --heads=8 --kfac_damping=0.1 --lr=0.01 --cov_ema_decay=0.25 \
    --baseline=adam --dataset=$dataset --runs=$VELICKOVIC_RUNS --file_name=results/GAT-$dataset-Veličković

