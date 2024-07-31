import argparse
import gc
import os
import time
from os.path import join, dirname

import matplotlib
import torch
from torch import Tensor, no_grad, tensor, nn
from torch.nn import Module, Linear, ModuleList
from torch.optim import Adam, SGD, AdamW
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import BatchNorm

import torch_geometric.transforms as T
import torch.nn.functional as F
from tqdm.auto import tqdm

import MFAC.optim
from examples.filter_blocks import FilterBlocksFactory
from plot_utils import plot_training, find_filename
from gat_conv import GATConv
import json

from gcn_conv import GCNConv
from torch_kfac import KFAC
from torch_kfac.layers import FullyConnectedFisherBlock, Bias, BiasFisherBlock, Identity
from hessianfree.optimizer import HessianFree

from torch_kfac.layers.fisher_block_factory import FisherBlockFactory

font = {'size': 18}
matplotlib.rc('font', **font)

def calculate_sparsity(grad: Tensor):
    # torch.nonzero returns the indices of the entries in the tensor, which are nonzero.
    # num_non_zero is the count of nonzero elements in "grad".
    num_non_zero = torch.nonzero(grad).size(0)
    total_elements = grad.numel()
    sparsity_ratio = 1 - (num_non_zero / total_elements)
    sparsity_norm = grad.norm().item()
    return sparsity_ratio, sparsity_norm


def print_sparsity_ratio(grad: Tensor, caption):
    sparsity_ratio, sparsity_norm = calculate_sparsity(grad)
    print(f"{caption} sparsity ratio: {sparsity_ratio*100:.3f}%")
    print(f"{caption} sparsity (norm): {sparsity_norm:.3f}")


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout,
                 hidden_layers=1, use_heads_last_channel=False,
                 add_self_loops=True):
        super().__init__()
        self.convs = ModuleList()
        for _ in range(hidden_layers):
            self.convs.append(GATConv(in_channels, hidden_channels, heads, dropout=dropout,
                                      add_self_loops=add_self_loops))
            in_channels = hidden_channels * heads

        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv_last = self.conv_last = GATConv(in_channels, out_channels, heads=heads if use_heads_last_channel else 1,
                                 concat=False, dropout=dropout, add_self_loops=add_self_loops)
        self.batch_norm = BatchNorm(hidden_channels * heads)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv_last.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_last(x, edge_index)
        #x = F.softmax(x, dim=1)
        return x

# The modules CRD, CLS, Net are from
# https://github.com/russellizadi/ssp/blob/master/experiments/gcn.py.

class CRD(Module):
    def __init__(self, d_in, d_out, p, add_self_loops=True):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True,
                            add_self_loops=add_self_loops)
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x


class CLS(Module):
    def __init__(self, d_in, d_out, add_self_loops=True):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True, add_self_loops=add_self_loops)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        return x


class GCN(Module):
    def __init__(self, dataset, hidden_layers: int = 1, hidden_dim: int = 16,
                 dropout: float = 0.5, add_self_loops=True):
        super(GCN, self).__init__()
        in_features = dataset.num_features
        self.crds = ModuleList()
        for i in range(hidden_layers):
            self.crds.append(CRD(in_features, hidden_dim, dropout, add_self_loops=add_self_loops))
            in_features = hidden_dim

        self.cls = CLS(in_features, dataset.num_classes, add_self_loops=add_self_loops)

    def reset_parameters(self):
        for crd in self.crds:
            crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for crd in self.crds:
            x = crd(x, edge_index, data.train_mask)
        x = self.cls(x, edge_index, data.train_mask)
        return x


# Functions train, evaluate and main are based on
# https://github.com/russellizadi/ssp/blob/master/experiments/train_eval.py.

def model_forward(model):
    model.eval()
    out = model(data)
    #loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], reduction="mean")
    return loss, out

def train(model, optimizer, data, preconditioner, update_cov):
    model.train()
    optimizer.zero_grad()

    loss = None
    if isinstance(preconditioner, KFAC):
        with preconditioner.track_forward():
            out = model(data)
            label = out.max(1)[1]
            label[data.train_mask] = data.y[data.train_mask]
            label.requires_grad = False

            loss = F.cross_entropy(out[data.train_mask], label[data.train_mask], reduction="mean")
        with preconditioner.track_backward():
            loss.backward(retain_graph=True)

        if update_cov:
            preconditioner.update_cov()
        preconditioner.step(loss)
    else:
        out = model(data)
        label = out.max(1)[1]
        label[data.train_mask] = data.y[data.train_mask]

        loss = F.cross_entropy(out[data.train_mask], label[data.train_mask], reduction="mean")
        loss.backward(retain_graph=True)
        if isinstance(preconditioner, HessianFree):
            model.eval() # need to set to eval mode otherwise model is not deterministic.
            preconditioner.step(lambda: model_forward(model), test_deterministic=True)
            model.train()
        if isinstance(preconditioner, MFAC.optim.MFAC):
            preconditioner.step()

    optimizer.step()


def evaluate(model, data):
    model.eval()

    with no_grad():
        logits = model(data)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.cross_entropy(logits[mask], data.y[mask], reduction="mean").item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{} loss'.format(key)] = loss
        outs['{} acc'.format(key)] = acc

    return outs


def train_model(dataset, args: argparse.Namespace, device):
    """
    Trains a model with a planetoid dataset.
    Args:
        dataset: the dataset which is used for training
        args: the arguments for the model and the training.
    """
    data = dataset[0].to(device)
    enable_kfac = args.enable_kfac
    weight_decay = args.weight_decay
    if args.model == "GCN":
        model = GCN(dataset, hidden_layers=args.hidden_layers,
                    hidden_dim=args.hidden_channels, dropout=args.dropout,
                    add_self_loops=args.add_self_loops)
        parameters = [dict(params=model.crds.parameters(), weight_decay=weight_decay),
                      dict(params=model.cls.parameters(), weight_decay=0)]
    elif args.model == "GAT":
        model = GAT(dataset.num_features, args.hidden_channels,
                    dataset.num_classes, args.heads, args.dropout,
                    args.hidden_layers, use_heads_last_channel=args.dataset == "PubMed",
                    add_self_loops=args.add_self_loops)
        parameters = [dict(params=model.convs.parameters(), weight_decay=weight_decay),
                      dict(params=model.conv_last.parameters(), weight_decay=0)]
    lr = args.lr

    kfac_lr = args.kfac_lr

    model.to(device).reset_parameters()
    optimizer_lr = kfac_lr if enable_kfac else lr
    optimizer = Adam(parameters, lr=optimizer_lr, weight_decay=weight_decay)
    #optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    preconditioner = None

    if enable_kfac:
        # damping corresponds to eps in PSGD.
        # cov_ema_decay corresponds to 1-alpha in PSGD.
        # PSGD does not implement momentum.
        #factory = None
        preconditioner = KFAC(model, 0.0, args.kfac_damping, momentum=0.0, damping_adaptation_decay=0.99,
                              cov_ema_decay=args.cov_ema_decay, enable_pi_correction=True, adapt_damping=True,
                              damping_adaptation_interval=5, update_cov_manually=True,
                              block_factory=FilterBlocksFactory)
        for block in preconditioner.blocks:
            if not isinstance(block, Identity):
                block.train_mask = data["train_mask"]

        linear_blocks = sum(1 for block in preconditioner.blocks if not isinstance(block, Identity))
        print(f"Preconditioning active on {linear_blocks} blocks.")
    elif args.baseline in ["Hessian", "GGN"] and not enable_kfac:
        preconditioner = HessianFree(model.parameters(), verbose=False, curvature_opt=args.baseline, cg_max_iter=1000,
                                lr=0.0, damping=args.hessianfree_damping, adapt_damping=args.baseline == "Hessian")
    elif args.baseline == "M-FAC" and not enable_kfac:
        from MFAC.optim import MFAC
        preconditioner = MFAC(model.parameters(), lr=0.0, moddev=device, optdev=device)

    if preconditioner is not None:
        print(f"Preconditioner: {preconditioner.__class__}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    epochs_tqdm = tqdm(range(1, args.epochs + 1))
    losses, val_accuracies, test_accuracies, train_accuracies = [], [], [], []
    for epoch in epochs_tqdm:
        train(model, optimizer, data, preconditioner, (epoch-1) % args.cov_update_freq == 0)
        eval_info = evaluate(model, data)
        epochs_tqdm.set_description(
            f"Val loss:{eval_info['val loss']:.2f}, Train loss: {eval_info['train loss']:.3f}, Val acc: {eval_info['val acc']:.3f}, Test acc: {eval_info['test acc']:.3f}")
        losses.append(eval_info['train loss'])
        val_accuracies.append(eval_info['val acc'] * 100)
        test_accuracies.append(eval_info['test acc'] * 100)
        train_accuracies.append(eval_info['train acc'] * 100)

    return losses, val_accuracies, test_accuracies, train_accuracies

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--model', type=str, required=True, choices=["GAT", "GCN"])
    parser.add_argument('--baseline', choices=['ADAM', 'Hessian', 'GGN', 'M-FAC'], default='ADAM')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--hidden_layers', type=int, default=1)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--kfac_lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--kfac_damping', type=float, default=0.1)
    parser.add_argument('--hessianfree_damping', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--cov_update_freq', type=int, default=1)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--cov_ema_decay', type=float, default=0.0)
    parser.add_argument('--results_dir', type=str, default="results")
    parser.add_argument('--file_name', type=str, default=None)
    parser.add_argument('--add_self_loops', type=bool, default=True)
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    if args.kfac_lr is None:
        args.kfac_lr = args.lr

    experiment_metadata = {
        "lr": args.lr,
        "kfac_lr": args.kfac_lr,
        "weight_decay": args.weight_decay,
        "dataset": args.dataset,
        "hidden_channels": args.hidden_channels,
        "layers": args.hidden_layers + 1,
        "runs": args.runs,
        "dropout": args.dropout,
        "kfac_damping": args.kfac_damping,
        "hessianfree_damping": args.hessianfree_damping,
        "heads": args.heads,
        "cov_ema_decay": args.cov_ema_decay,
        "cov_update_freq": args.cov_update_freq,
        "epochs": args.epochs
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    name = args.dataset
    dataset = Planetoid(join(dirname(__file__), '..', 'data', name), name)
    dataset.transform = T.NormalizeFeatures()
    data = dataset[0].to(device)

    epochs = args.epochs
    gamma = None

    def train_fun(params):
        return train_model(dataset, argparse.Namespace(**params), device)

    _, _, fig, group_results = plot_training(args.epochs, args.runs, train_fun, [
        {**vars(args), "enable_kfac": False}, {**vars(args), "enable_kfac": True},
    ], [args.baseline, "KFAC"], loss_range=(0,2))

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
