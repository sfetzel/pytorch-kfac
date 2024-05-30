import argparse
import gc
import os.path as osp
import time
import os

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from matplotlib import pyplot
from torch_geometric.datasets import Planetoid, OGB_MAG
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import BatchNorm
from torch import Tensor, mean, std
import itertools
from joblib import Parallel, delayed

from gat_conv import GATConv
from tqdm.auto import tqdm
from planetoid import seed_everything
from torch_kfac import KFAC
from torch_kfac.layers import FullyConnectedFisherBlock
from gcn_conv import GCNConv
from torch.nn import Module, ModuleList

class CRD(Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x


class CLS(Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


class GCN(Module):
    def __init__(self, dataset, hidden_layers: int = 1, hidden_dim: int = 16,
                 dropout=0.5):
        super(GCN, self).__init__()
        in_features = dataset.num_features
        self.crds = ModuleList()
        for i in range(hidden_layers):
            self.crds.append(CRD(in_features, hidden_dim, dropout))
            in_features = hidden_dim

        self.cls = CLS(in_features, dataset.num_classes)

    def reset_parameters(self):
        for crd in self.crds:
            crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, x, edge_index):
        #x, edge_index = data.x, data.edge_index
        train_mask = None #data.train_mask
        for crd in self.crds:
            x = crd(x, edge_index)
        x = self.cls(x, edge_index)
        return x



class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=dropout)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=dropout)
        self.batch_norm = BatchNorm(hidden_channels * heads)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        #x = self.batch_norm(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CiteSeer')
    parser.add_argument('--model', type=str, choices=['GAT', 'GCN'], required=True)
    parser.add_argument('--hidden_channels', type=int, default=8)
    parser.add_argument('--hidden_layers', type=int, default=1)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--dropouts', nargs='+', type=float, default=[0.6])
    parser.add_argument('--lrs', nargs='+', type=float, default=[0.1, 0.01])
    parser.add_argument('--kfac_dampings', nargs='+', type=float, default=[0.1, 0.01])
    parser.add_argument('--cov_ema_decays', nargs='+', type=float, default=[0.0, 0.85])
    parser.add_argument('--weight_decays', nargs='+', type=float, default=[0.0005])
    parser.add_argument('--cov_update_freqs', nargs='+', type=int, default=[1])
    parser.add_argument('--results_dir', type=str, default="results")
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    #path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'OgbMag')
    #dataset = OGB_MAG(path)
    data = dataset[0].to(device)


def train(model, optimizer, preconditioner, update_cov):
    model.train()
    optimizer.zero_grad()
    if isinstance(preconditioner, KFAC):
        with preconditioner.track_forward():
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        with preconditioner.track_backward():
            loss.backward()
    else:
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

    train_nodes = sum(data.train_mask)
    if isinstance(preconditioner, KFAC):
        if update_cov:
            fishy_stuff = True
            if fishy_stuff:
                sensitivities_shapes = []
                activations_shapes = []
                for block in preconditioner.blocks:
                    if isinstance(block, FullyConnectedFisherBlock):
                        # PSGD doesn't filter the sensitivities or activations.
                        # block._activations = block._activations[data.train_mask]
                        # block._activations = block._activations[data.train_mask]

                        # difference: PSGD multiplies gradient in backward hook with shape[1] instead of shape[0].
                        block._sensitivities = block._sensitivities / block._sensitivities.shape[0] * \
                                               block._sensitivities.shape[1]
                        sensitivities_shapes.append(block._sensitivities.shape)
                        activations_shapes.append(block._activations.shape)

            preconditioner.update_cov()
            if fishy_stuff:
                i = 0
                for block in preconditioner.blocks:
                    if isinstance(block, FullyConnectedFisherBlock):
                        # The PSGD implementation does not divide by count of nodes, but by count of training nodes.
                        block._activations_cov.value = block._activations_cov.value * activations_shapes[i][0] / train_nodes
                        block._sensitivities_cov.value = block._sensitivities_cov.value * sensitivities_shapes[i][
                            0] / train_nodes
                        i += 1

    preconditioner.step(loss)
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


def train_model(dropout, kfac_damping, lr, weight_decay, cov_ema_decay, cov_update_freq):
    if args.model == "GAT":
        model = GAT(dataset.num_features, args.hidden_channels, dataset.num_classes,
                    args.heads, dropout).to(device)
    elif args.model == "GCN":
        model = GCN(dataset, hidden_dim=args.hidden_channels, dropout=dropout,
                    hidden_layers=args.hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    preconditioner = KFAC(model, 0, kfac_damping **0.5, cov_ema_decay=cov_ema_decay, adapt_damping=False, damping_adaptation_decay=0.99,
                          enable_pi_correction=False, update_cov_manually=True, momentum=0.0,
                          damping_adaptation_interval=5)
    linear_blocks = sum(1 for block in preconditioner.blocks if isinstance(block, FullyConnectedFisherBlock))
    print(f"Preconditioning active on {linear_blocks} blocks.")

    best_results = None
    epochs_tqdm = tqdm(range(args.epochs))
    for epoch in epochs_tqdm:
        loss = train(model, optimizer, preconditioner, epoch % cov_update_freq == 0)
        accs = test(model)
        if best_results is None or best_results[1] < accs[1]:
            best_results = accs
            best_results.append(loss)
        epochs_tqdm.set_description(f"Loss: {loss:.3f}, Val acc: {accs[1]*100:.2f} (best: {best_results[1]*100:.2f} with test acc:{best_results[2]*100:.2f})")

    return best_results

if __name__ == "__main__":
    parameters = [args.dropouts, args.kfac_dampings, args.lrs, args.weight_decays, args.cov_ema_decays, args.cov_update_freqs]
    seed_everything()
    file_path = os.path.join(args.results_dir, f"{args.model}_{args.dataset}_gridsearch.csv")
    print(file_path)
    with open(file_path, "w+") as results_file:
        results_file.write("dropout;kfac_damping;lr;weight_decay;cov_update_freq;cov_ema_decay;train_acc_mean;train_acc_std;val_acc_mean;val_acc_std;test_acc_mean;test_acc_std;loss_mean;loss_std\n")
        parameters_list = itertools.product(*parameters)

        for params in parameters_list:
            dropout, kfac_damping, lr, weight_decay, ema_decay, cov_update_freq = params

            all_results = []
            for _ in range(args.runs):
                all_results.append(train_model(dropout, kfac_damping, lr, weight_decay, ema_decay, cov_update_freq))
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



