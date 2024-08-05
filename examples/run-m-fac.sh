
#!/bin/bash

# you need to clone https://github.com/sfetzel/m-fac into the folder MFAC first.
# git clone https://github.com/sfetzel/M-FAC.git MFAC

PLANETOID_DATASETS="Cora CiteSeer PubMed"

RUNS=10
for dataset in $PLANETOID_DATASETS; do
  for model in GCN GAT; do
    python planetoid.py --model=$model --baseline=M-FAC --mfac_damping=0.1 --dataset=$dataset --runs=$RUNS --file_name=results/$model-$dataset-m-fac
    python planetoid.py --cov_update_freq=50 --model=$model --baseline=M-FAC --mfac_damping=0.1 --dataset=$dataset --runs=$RUNS --file_name=results/$model-$dataset-m-fac-reduced-f
  done
done
