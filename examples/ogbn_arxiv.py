# based on
# https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py
import argparse

import torch
from torch import no_grad
import torch.nn.functional as F
import torch_geometric.nn.dense

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, #GATConv #, GCNConv
from torch_geometric.nn.dense import Linear
from torch_sparse import SparseTensor
from torch import Tensor
from gcn_conv import GCNConv
from gat_conv import GATConv
from filter_blocks import FilterBlocksFactory
import os
import json
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch

from torch_kfac import KFAC
from torch_kfac.layers import Identity
from plot_utils import plot_training, find_filename

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')



class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)



class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, num_layers,
                 dropout):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        # ToDo: update GATConv and readd cached=True.
        self.convs.append(GATConv(in_channels, hidden_channels, heads))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, 1))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)



def train(model, data, train_idx, optimizer, preconditioner, enable_kfac, update_cov):
    model.train()

    optimizer.zero_grad()
    if enable_kfac:
        with preconditioner.track_forward():
            out = model(data.x, data.adj_t)[train_idx]
            loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
        with preconditioner.track_backward():
            loss.backward()

        if update_cov:
            preconditioner.update_cov()
        preconditioner.step(loss)
    else:
        torch.autograd.set_detect_anomaly(True)
        out = model(data.x, data.adj_t)[train_idx]
        loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
        loss.backward()

    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--baseline', type=str, choices=["adam"], default="adam")
    parser.add_argument('--model', type=str, choices=["GCN", "SAGE", "GAT"], default="GCN")
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--kfac_lr', type=float, default=None)
    parser.add_argument('--kfac_damping', type=float, default=2e-7)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--cov_ema_decay', type=float, default=0.0)
    parser.add_argument('--results_dir', type=str, default="results")
    parser.add_argument('--file_name', type=str, default=None)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--cov_update_freq', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    args = parser.parse_args()

    if args.kfac_lr is None:
        args.kfac_lr = args.lr

    print(args)
    dataset_name = 'ogbn-arxiv'

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    experiment_metadata = {
        "lr": args.lr,
        "kfac_lr": args.kfac_lr,
        "weight_decay": args.weight_decay,
        "dataset": dataset_name,
        "hidden_channels": args.hidden_channels,
        "layers": args.num_layers,
        "runs": args.runs,
        "dropout": args.dropout,
        "kfac_damping": args.kfac_damping,
        "heads": args.heads,
        "cov_ema_decay": args.cov_ema_decay,
        "cov_update_freq": args.cov_update_freq
    }
    device = f'{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dataset = PygNodePropPredDataset(name=dataset_name,
                                     transform=T.Compose([T.ToUndirected()]))
    #,T.ToSparseTensor()


    data = dataset[0]
    data.adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1])
    data.adj_t = data.adj.t()
    # use T.ToUndirected() instead of to_symmetric
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    factory = FilterBlocksFactory

def train_model(args, device):
    enable_kfac = args.enable_kfac
    if args.model == "SAGE":
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
        parameters = [dict(params=model.convs[:-1].parameters(), weight_decay=args.weight_decay),
                      dict(params=model.convs[-1].parameters(), weight_decay=0)]
    elif args.model == "GCN":
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)
        parameters = [dict(params=model.convs[:-1].parameters(), weight_decay=args.weight_decay),
                      dict(params=model.convs[-1].parameters(), weight_decay=0)]
    if args.model == "GAT":
        model = GAT(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.heads, args.num_layers,
                     args.dropout).to(device)
        parameters = [dict(params=model.convs[:-1].parameters(), weight_decay=args.weight_decay),
                      dict(params=model.convs[-1].parameters(), weight_decay=0)]
    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    model.reset_parameters()
    optimizer_lr = args.kfac_lr if enable_kfac else args.lr
    optimizer = torch.optim.Adam(parameters, lr=optimizer_lr)
    preconditioner = KFAC(model, 0.0, args.kfac_damping,
                          cov_ema_decay=args.cov_ema_decay, adapt_damping=True,
                          update_cov_manually=True,
                          momentum=0.0, block_factory=factory)
    if enable_kfac:
        preconditioning_blocks = 0
        for block in preconditioner.blocks:
            if not isinstance(block, Identity):
                block.train_mask = train_idx
                preconditioning_blocks += 1
        print(f"{preconditioning_blocks} preconditioning blocks")

    losses, val_accuracies, test_accuracies, train_accuracies = [], [], [], []
    for epoch in range(1, 1 + args.epochs):
        loss = train(model, data, train_idx, optimizer, preconditioner, enable_kfac, epoch % args.cov_update_freq == 0)
        result = test(model, data, split_idx, evaluator)

        if epoch % args.log_steps == 0:
            train_acc, valid_acc, test_acc = result
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')
        losses.append(loss)
        val_accuracies.append(valid_acc * 100)
        test_accuracies.append(test_acc * 100)
        train_accuracies.append(train_acc * 100)

    return losses, val_accuracies, test_accuracies, train_accuracies


if __name__ == "__main__":
    def plot_training_train_model(params):
        merged_args = {**vars(args), "enable_kfac": params["kfac"]}
        args_copy = argparse.Namespace(**merged_args)
        return train_model(args_copy, device)


    _, _, fig, group_results = plot_training(args.epochs, args.runs, plot_training_train_model, [
        {"kfac": False}, {"kfac": True},
    ], [args.baseline, "KFAC"],loss_range=(0, 4.0), legend_loc='upper right')

    fig.suptitle(f"{args.model} Training on {dataset_name}", fontsize=28)
    filename = args.file_name
    if filename is None:
        filename = find_filename(os.path.join(args.results_dir, f"{args.model}-{dataset_name}"), "svg")
    print(filename)
    fig.savefig(f"{filename}.svg", metadata={
        "Description": f"Experiment properties: {str(experiment_metadata)}, Grouped results: {str(group_results)}"
    })

    with open(f"{filename}.txt", "w") as metadata_file:
        metadata_file.write(json.dumps({
            "experiment": experiment_metadata,
            "results": group_results
        }, indent=4))

