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

python3 comparison.py --device=cuda:1 --epochs=200 --dataset_name=PROTEINS --model=GIN > comparison-gin-proteins.txt;
python3 comparison.py --device=cuda:1 --epochs=200 --dataset_name=COLLAB --model=GIN > comparison-gin-collab.txt;
python comparison.py --device=cuda:1 --epochs=200 --dataset_name=REDDIT-BINARY --model=GIN > comparison-gin-reddit-binary.txt
python comparison.py --device=cuda:1 --epochs=200 --dataset_name=DD --model=GIN > comparison-gin-dd.txt


RUNS=10
GRIDSEARCH_RUNS=5
python gridsearch.py --epochs=200 --model=GCN --hidden_channels=64 --dataset=Cora --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 0.85 --dropouts 0.5 0.6 --cov_update_freqs 1 --lrs 0.01 --kfac_dampings 0.1 0.001 --weight_decays 0.0005
python gridsearch.py --epochs=200 --model=GCN --hidden_channels=64 --dataset=CiteSeer --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 0.85 --dropouts 0.5 --cov_update_freqs 25 50 1 --lrs 0.01 --kfac_dampings 0.1 0.01 --weight_decays 0.01 0.05 0.005

#python gridsearch.py --epochs=300 --model=GCN --dataset=CiteSeer --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 --lrs 0.1 0.005 --kfac_dampings 0.1 0.01 0.001
#python gridsearch.py --epochs=300 --model=GCN --dataset=PubMed --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 --lrs 0.1 0.005 --kfac_dampings 0.1 0.01 0.001
#python gridsearch.py --epochs=300 --model=GAT --dataset=Cora --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 --lrs 0.1 0.005 --kfac_dampings 0.1 0.01 0.001
#python gridsearch.py --epochs=300 --model=GAT --dataset=CiteSeer --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 --lrs 0.1 0.005 --kfac_dampings 0.1 0.01 0.001
#python gridsearch.py --epochs=300 --model=GAT --dataset=PubMed --runs=$GRIDSEARCH_RUNS --cov_ema_decays 0.0 0.25 --lrs 0.1 0.005 --kfac_dampings 0.1 0.01 0.001



RUNS=10

for baseline in adam hessian ggn; do
  python planetoid.py --model=GCN --dropout=0.6 --kfac_damping=0.1 --lr=0.01 --cov_ema_decay=0.0 --baseline=$baseline --dataset=Cora --runs=$RUNS
done

dataset=Cora
for damping in 0.1 0.001 0.0001; do
  python planetoid.py --cov_update_freq=1 --model=GCN --dropout=0.5 --weight_decay=5e-4\
    --hidden_channels=64 --kfac_damping=$damping --lr=0.01 --cov_ema_decay=0.0 \
    --baseline=adam --dataset=$dataset --runs=10 --file_name=results/GCN-$dataset-damping-$damping
done

for cov_ema_decay in "0.0" "0.25" "0.5" "0.85"; do
  python planetoid.py --cov_update_freq=1 --model=GCN --dropout=0.5 --weight_decay=5e-4\
    --hidden_channels=64 --kfac_damping=0.01 --lr=0.01 --cov_ema_decay=$cov_ema_decay \
    --baseline=adam --dataset=$dataset --runs=10 --file_name=results/GCN-$dataset-cov-decay-$cov_ema_decay
done

python ogbn_arxiv.py --device=1 --model=GCN --runs=100 --file_name=results/GCN-ogbn-arxiv
# not enough memory?
python ogbn_arxiv.py --device=cpu --model=GAT --hidden_channels=32 --heads=8 --runs=10