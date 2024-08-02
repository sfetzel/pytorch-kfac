#!/bin/bash
DAMPING="0.1 None"
DEVICE=cuda:0
EPOCHS=200
PATIENCE=500

python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=PROTEINS --model=GIN > comparison-gin-proteins.txt;
python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=ENZYMES --model=GIN > comparison-gin-enzymes.txt;

python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=MUTAG --model=GIN > comparison-gin-mutag.txt;
python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=COLLAB --model=GIN > comparison-gin-collab.txt;
python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=IMDB-BINARY --model=GIN > comparison-gin-imdb-binary.txt;
python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=IMDB-MULTI --model=GIN > comparison-gin-imdb-multi.txt;
python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=DD --model=GIN > comparison-gin-dd.txt
python comparison.py --kfac_damping $DAMPING --device=$DEVICE --layer 5  --epochs=$EPOCHS --patience $PATIENCE --dataset_name=REDDIT-BINARY --model=GIN > comparison-gin-reddit-binary-5-layers.txt


python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=PROTEINS --model=GAT > comparison-gat-proteins.txt;
python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=ENZYMES --model=GAT > comparison-gat-enzymes.txt;
python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=MUTAG --model=GAT > comparison-gat-mutag.txt;
#python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=COLLAB --model=GAT > comparison-gat-collab.txt;
#python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=IMDB-BINARY --model=GAT > comparison-gat-imdb-binary.txt;
#python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=IMDB-MULTI --model=GAT > comparison-gat-imdb-multi.txt;
python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=DD --model=GAT > comparison-gat-dd.txt

#python comparison.py --kfac_damping $DAMPING --device=$DEVICE --epochs=$EPOCHS --patience $PATIENCE --dataset_name=REDDIT-BINARY --model=GAT > comparison-gat-reddit-binary.txt

