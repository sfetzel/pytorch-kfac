#!/bin/bash
DAMPING="0.1 None"
DEVICE=cuda:0
EPOCHS=200
PATIENCE=500

model="GIN"
python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=PROTEINS --model=$model > comparison-$model-proteins.txt;
python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=ENZYMES --model=$model > comparison-$model-enzymes.txt;

python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=MUTAG --model_selection_metric=loss --model=$model > comparison-$model-mutag.txt;
python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=COLLAB --model=$model > comparison-$model-collab.txt;
python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=IMDB-BINARY --model=$model > comparison-$model-imdb-binary.txt;
python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=IMDB-MULTI --model=$model > comparison-$model-imdb-multi.txt;
python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=DD --model=$model > comparison-$model-dd.txt
python comparison.py --kfac_damping $DAMPING --device=$DEVICE --layer 5  --epochs=$EPOCHS --patience $PATIENCE --dataset_name=REDDIT-BINARY --model=$model > comparison-$model-reddit-binary-5-layers.txt

model="GIN-simple"
python comparison.py --aggregation sum --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=PROTEINS --model=$model > comparison-$model-proteins.txt;
python comparison.py --aggregation sum --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=ENZYMES --model=$model > comparison-$model-enzymes.txt;

python comparison.py --aggregation sum --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=MUTAG --model_selection_metric=loss --model=$model > comparison-$model-mutag.txt;
python comparison.py --aggregation sum --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=COLLAB --model=$model > comparison-$model-collab.txt;
python comparison.py --aggregation sum --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=IMDB-BINARY --model=$model > comparison-$model-imdb-binary.txt;
python comparison.py --aggregation sum --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=IMDB-MULTI --model=$model > comparison-$model-imdb-multi.txt;
python comparison.py --aggregation sum --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=DD --model=$model > comparison-$model-dd.txt
python comparison.py --aggregation sum --kfac_damping $DAMPING --device=$DEVICE --layer 5  --epochs=$EPOCHS --patience $PATIENCE --dataset_name=REDDIT-BINARY --model=$model > comparison-$model-reddit-binary-5-layers.txt


python comparison.py --aggregation sum --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=PROTEINS --model=GAT > comparison-gat-proteins.txt;
python comparison.py --aggregation sum --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=ENZYMES --model=GAT > comparison-gat-enzymes.txt;
python comparison.py --aggregation sum --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=MUTAG --model_selection_metric=loss --model=GAT > comparison-gat-mutag.txt;
#python comparison.py --aggregation sum --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=COLLAB --model=GAT > comparison-gat-collab.txt;
#python comparison.py --aggregation sum --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=IMDB-BINARY --model=GAT > comparison-gat-imdb-binary.txt;
#python comparison.py --aggregation sum --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=IMDB-MULTI --model=GAT > comparison-gat-imdb-multi.txt;
python comparison.py --aggregation sum --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=DD --model=GAT > comparison-gat-dd.txt

#python comparison.py --aggregation sum --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=REDDIT-BINARY --model=GAT > comparison-gat-reddit-binary.txt

