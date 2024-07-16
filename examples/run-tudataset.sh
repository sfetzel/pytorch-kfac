#!/bin/bash

python3 comparison.py --device=cuda:1 --epochs=200 --dataset_name=PROTEINS --model=GIN > comparison-gin-proteins.txt;
python3 comparison.py --device=cuda:1 --epochs=200 --dataset_name=COLLAB --model=GIN > comparison-gin-collab.txt;
python comparison.py --device=cuda:1 --epochs=200 --dataset_name=REDDIT-BINARY --model=GIN > comparison-gin-reddit-binary.txt
python comparison.py --device=cuda:1 --epochs=200 --dataset_name=DD --model=GIN > comparison-gin-dd.txt
