import argparse
import gc
import os.path as osp
import time
import os

import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, OGB_MAG
from torch import Tensor, mean, std
import itertools

from planetoid import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CiteSeer')
    parser.add_argument('--model', type=str, choices=['GAT', 'GCN'], required=True)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--hidden_layers', type=int, default=1)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--dropouts', nargs='+', type=float, default=[0.4, 0.5, 0.6])
    parser.add_argument('--lrs', nargs='+', type=float, default=[0.01])
    parser.add_argument('--kfac_dampings', nargs='+', type=float, default=[0.1])
    parser.add_argument('--cov_ema_decays', nargs='+', type=float, default=[0.0])
    parser.add_argument('--weight_decays', nargs='+', type=float, default=[0.05, 0.005, 0.0005])
    parser.add_argument('--cov_update_freqs', nargs='+', type=int, default=[1])
    parser.add_argument('--results_dir', type=str, default="results")
    parser.add_argument('--add_self_loops', type=bool, default=True)
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    parameters = [args.dropouts, args.kfac_dampings, args.lrs, args.weight_decays, args.cov_ema_decays, args.cov_update_freqs]

    file_path = os.path.join(args.results_dir, f"{args.model}_{args.dataset}_gridsearch.csv")
    print(file_path)
    with open(file_path, "w+") as results_file:
        results_file.write("dropout;kfac_damping;lr;weight_decay;cov_update_freq;cov_ema_decay;train_acc_mean;train_acc_std;val_acc_mean;val_acc_std;test_acc_mean;test_acc_std;loss_mean;loss_std\n")
        parameters_list = itertools.product(*parameters)

        for params in parameters_list:
            dropout, kfac_damping, lr, weight_decay, ema_decay, cov_update_freq = params

            all_results = []
            for _ in range(args.runs):
                merged_args = {**vars(args), "enable_kfac": True,
                               "dropout": dropout, "kfac_damping": kfac_damping, "lr": lr,
                               "weight_decay": weight_decay, "cov_ema_decay": ema_decay,
                               "cov_update_freq": cov_update_freq,
                               "kfac_lr": lr}
                args_copy = argparse.Namespace(**merged_args)
                losses, val_accuracies, test_accuracies, train_accuracies = train_model(dataset, args_copy, device)
                best_epoch = torch.argmax(Tensor(val_accuracies))
                all_results.append([train_accuracies[best_epoch], val_accuracies[best_epoch],
                                    test_accuracies[best_epoch], losses[best_epoch]])
                gc.collect()
            results_tensor = Tensor(all_results)
            results_mean = mean(results_tensor, dim=0)
            results_std = std(results_tensor, dim=0)
            results_file.write(f"{dropout};{kfac_damping};{lr};{weight_decay};{cov_update_freq};{ema_decay};" +
                               f"{results_mean[0]};{results_std[0]};" +
                               f"{results_mean[1]};{results_std[1]};" +
                               f"{results_mean[2]};{results_std[2]};"
                               f"{results_mean[3]};{results_std[3]}\n")
            print(params)
            print(results_mean)
            results_file.flush()



