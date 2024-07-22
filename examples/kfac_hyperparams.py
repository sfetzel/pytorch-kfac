import argparse
import gc
import os
from os.path import join, dirname
import torch
from torch_geometric.datasets import Planetoid

import torch_geometric.transforms as T

from plot_utils import plot_training, find_filename
import json

from planetoid import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--model', type=str, required=True, choices=["GAT", "GCN"])
    parser.add_argument('--experiment', type=str, required=True, choices=["decay", "damping", "update_freq"])
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--hidden_layers', type=int, default=1)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--kfac_lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--results_dir', type=str, default="results")
    parser.add_argument('--file_name', type=str, default=None)
    parser.add_argument("--enable_kfac", type=bool, default=True)
    parser.add_argument("--add_self_loops", type=bool, default=True)
    args = parser.parse_args()

    if args.kfac_lr is None:
        args.kfac_lr = args.lr

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    experiment_metadata = {
        "lr": args.lr,
        "kfac_lr": args.lr,
        "weight_decay": args.weight_decay,
        "dataset": args.dataset,
        "hidden_channels": args.hidden_channels,
        "layers": args.hidden_layers + 1,
        "runs": args.runs,
        "dropout": args.dropout,
        "heads": args.heads,
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    name = args.dataset
    dataset = Planetoid(join(dirname(__file__), '..', 'data', name), name)
    dataset.transform = T.NormalizeFeatures()
    data = dataset[0].to(device)

    epochs = args.epochs
    gamma = None
    runs = 1

    def plot_training_train_model(params):
        merged_args = {**vars(args), "kfac_damping": params["kfac_damping"],
                       "cov_ema_decay": params["cov_ema_decay"],
                       "cov_update_freq": params["cov_update_freq"]}
        args_copy = argparse.Namespace(**merged_args)
        return train_model(dataset, args_copy, device)

    if args.experiment == "decay":
        _, _, fig, group_results = plot_training(args.epochs, args.runs, plot_training_train_model, [
            {"cov_ema_decay": 0.0, "kfac_damping": args.kfac_damping, "cov_update_freq": 1,},
            {"cov_ema_decay": 0.2, "kfac_damping": args.kfac_damping, "cov_update_freq": 1},
            {"cov_ema_decay": 0.5, "kfac_damping": args.kfac_damping, "cov_update_freq": 1},
        ], ['$\\gamma=0.1$', '$\\gamma=0.01$', '$\\gamma=0.001$'], loss_range=(0, 2))
    elif args.experiment == "damping":
        _, _, fig, group_results = plot_training(args.epochs, args.runs, plot_training_train_model, [
            {"cov_ema_decay": args.cov_ema_decay, "kfac_damping": 0.1, "cov_update_freq": 1,},
            {"cov_ema_decay": args.cov_ema_decay, "kfac_damping": 0.01, "cov_update_freq": 1,},
            {"cov_ema_decay": args.cov_ema_decay, "kfac_damping": 0.001, "cov_update_freq": 1,},
        ], ['$\\beta=0.1$', '$\\beta=0.01$', '$\\beta=0.001$'], loss_range=(0, 2))
    elif args.experiment == "update_freq":
        _, _, fig, group_results = plot_training(args.epochs, args.runs, plot_training_train_model, [
            {"cov_ema_decay": 0.0, "kfac_damping": 0.1, "cov_update_freq": 1,},
            {"cov_ema_decay": 0.0, "kfac_damping": 0.1, "cov_update_freq": 25,},
            {"cov_ema_decay": 0.0, "kfac_damping": 0.1, "cov_update_freq": 50,},
        ], ['$f=1$', '$f=25$', '$f=50$'], loss_range=(0, 2))
    fig.suptitle(f"{args.model} Training on {args.dataset}", fontsize=28)
    filename = args.file_name
    if filename is None:
        filename = find_filename(os.path.join(args.results_dir, f"{args.model}-{args.dataset}"), "svg")
    print(filename)
    fig.savefig(f"{filename}.svg", metadata={
        "Description": f"Experiment properties: {str(experiment_metadata)}, Grouped results: {str(group_results)}"
    })

    with open(f"{filename}.txt", "w") as metadata_file:
        metadata_file.write(json.dumps({
            "experiment": experiment_metadata,
            "results": group_results
        }, indent=4))
