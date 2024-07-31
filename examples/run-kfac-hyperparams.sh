#!/bin/bash
PLANETOID_DATASETS="Cora CiteSeer PubMed"

for dataset in $PLANETOID_DATASETS; do
  python kfac_hyperparams.py --model=GCN --dataset=$dataset --experiment=damping --runs=100 --file_name=results/GCN-$dataset-damping;
  python kfac_hyperparams.py --model=GCN --dataset=$dataset --experiment=decay --runs=100 --file_name=results/GCN-$dataset-decay;
  python kfac_hyperparams.py --model=GCN --dataset=$dataset --experiment=update_freq --runs=100 --file_name=results/GCN-$dataset-update-freq;
done
