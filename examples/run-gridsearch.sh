#!/bin/bash

GRIDSEARCH_RUNS=5
DATASETS="Cora CiteSeer PubMed"
COV_UPDATE_FREQ=50
for dataset in $DATASETS; do
  python gridsearch.py --cov_update_freqs $COV_UPDATE_FREQ --model=GCN --hidden_channels=64 --dataset=$dataset --runs=$GRIDSEARCH_RUNS
done
for dataset in $DATASETS; do
  python gridsearch.py --cov_update_freqs $COV_UPDATE_FREQ --model=GAT --hidden_channels=8 --heads=8 --dataset=$dataset --runs=$GRIDSEARCH_RUNS
done

#python gridsearch.py --epochs=300 --model=GCN --dataset=CiteSeer --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 --lrs 0.1 0.005 --kfac_dampings 0.1 0.01 0.001
#python gridsearch.py --epochs=300 --model=GCN --dataset=PubMed --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 --lrs 0.1 0.005 --kfac_dampings 0.1 0.01 0.001
#python gridsearch.py --epochs=300 --model=GAT --dataset=Cora --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 --lrs 0.1 0.005 --kfac_dampings 0.1 0.01 0.001
#python gridsearch.py --epochs=300 --model=GAT --dataset=CiteSeer --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 --lrs 0.1 0.005 --kfac_dampings 0.1 0.01 0.001
#python gridsearch.py --epochs=300 --model=GAT --dataset=PubMed --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 --lrs 0.1 0.005 --kfac_dampings 0.1 0.01 0.001
