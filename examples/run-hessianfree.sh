
RUNS=10
for dataset in $PLANETOID_DATASETS; do
  for model in GCN GAT; do
    python planetoid.py --model=$model --baseline=hessian --hessianfree_damping=1.0 --dataset=$dataset --runs=$RUNS --file_name=results/$model-$dataset-hessian
    python planetoid.py --model=$model --baseline=ggn --dataset=$dataset --runs=$RUNS --file_name=results/$model-$dataset-ggn
  done
done