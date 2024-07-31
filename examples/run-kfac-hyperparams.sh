#!/bin/bash
PLANETOID_DATASETS="Cora CiteSeer PubMed"
MODELS="GCN GAT"

for dataset in $PLANETOID_DATASETS; do
  for model in $MODELS; do
    python kfac_hyperparams.py --model=$model --dataset=$dataset --experiment=damping --runs=100 --file_name=results/$model-$dataset-damping;
    python kfac_hyperparams.py --model=$model --dataset=$dataset --experiment=decay --runs=100 --file_name=results/$model-$dataset-decay;
    python kfac_hyperparams.py --model=$model --dataset=$dataset --experiment=update_freq --runs=100 --file_name=results/$model-$dataset-update-freq;
  done
done
