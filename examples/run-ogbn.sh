#!/bin/sh

python ogbn_arxiv.py --device=1 --model=GCN --runs=100 --file_name=results/GCN-ogbn-arxiv
# not enough memory?
python ogbn_arxiv.py --device=cpu --model=GAT --hidden_channels=32 --heads=8 --runs=10 --file_name=results/GAT-ogbn-arxiv
