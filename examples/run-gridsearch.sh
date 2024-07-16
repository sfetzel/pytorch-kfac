#!/bin/bash

RUNS=10
GRIDSEARCH_RUNS=5
python gridsearch.py --epochs=200 --model=GCN --hidden_channels=64 --dataset=Cora --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 0.85 --dropouts 0.5 0.6 --cov_update_freqs 1 --lrs 0.01 --kfac_dampings 0.1 0.001 --weight_decays 0.0005
python gridsearch.py --epochs=200 --model=GCN --hidden_channels=64 --dataset=CiteSeer --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 0.85 --dropouts 0.5 --cov_update_freqs 25 50 1 --lrs 0.01 --kfac_dampings 0.1 0.01 --weight_decays 0.01 0.05 0.005

#python gridsearch.py --epochs=300 --model=GCN --dataset=CiteSeer --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 --lrs 0.1 0.005 --kfac_dampings 0.1 0.01 0.001
#python gridsearch.py --epochs=300 --model=GCN --dataset=PubMed --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 --lrs 0.1 0.005 --kfac_dampings 0.1 0.01 0.001
#python gridsearch.py --epochs=300 --model=GAT --dataset=Cora --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 --lrs 0.1 0.005 --kfac_dampings 0.1 0.01 0.001
#python gridsearch.py --epochs=300 --model=GAT --dataset=CiteSeer --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 --lrs 0.1 0.005 --kfac_dampings 0.1 0.01 0.001
#python gridsearch.py --epochs=300 --model=GAT --dataset=PubMed --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 --lrs 0.1 0.005 --kfac_dampings 0.1 0.01 0.001
