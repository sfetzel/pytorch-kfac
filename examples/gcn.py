import argparse
import gc
import os
import time
from os.path import join, dirname
import torch
from torch import Tensor, no_grad, tensor
from torch.nn import Module, Linear, ModuleList
from torch.optim import Adam, SGD, AdamW
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot
import torch_geometric.transforms as T
from torch_landscape.utils import seed_everything
from torch_sparse import SparseTensor, matmul
import torch.nn.functional as F
from tqdm.auto import tqdm
from plot_utils import plot_training

from torch_kfac import KFAC
from torch_kfac.layers import FullyConnectedFisherBlock
from hessianfree.optimizer import HessianFree


def calculate_sparsity(grad: Tensor):
    # torch.nonzero returns the indices of the entries in the tensor, which are nonzero.
    # num_non_zero is the count of nonzero elements in "grad".
    num_non_zero = torch.nonzero(grad).size(0)
    total_elements = grad.numel()
    sparsity_ratio = 1 - (num_non_zero / total_elements)
    sparsity_norm = grad.norm().item()
    return sparsity_ratio, sparsity_norm


def print_sparsity_ratio(grad: Tensor, caption: int):
    sparsity_ratio, sparsity_norm = calculate_sparsity(grad)
    print(f"{caption} sparsity ratio:", sparsity_ratio)
    print(f"{caption} sparsity (norm):", sparsity_norm)


class WeightedMessagePassing(MessagePassing):
    """
    Message passing which also considers the edge weights.
    """

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        r"""Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :obj:`edge_index` and applies multiplies the messages with the
        corresponding edge weight.
        """
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        r"""Fuses computations of :func:`message` and :func:`aggregate` into a
        single function.
        If applicable, this saves both time and memory since messages do not
        explicitly need to be materialized.
        This function will only gets called in case it is implemented and
        propagation takes place based on a :obj:`torch_sparse.SparseTensor`
        or a :obj:`torch.sparse.Tensor`.
        """
        return matmul(adj_t, x, reduce=self.aggr)


class GCNConv(Module):
    """
    A GCN Convolution layer which first executes message passing,
    then applies a linear transform.
    """

    def __init__(self, d_in, d_out, cached=True, bias=True):
        super(GCNConv, self).__init__()
        self.lin = Linear(d_in, d_out, bias=bias)
        self.bias = self.lin.bias
        self.message_passing = WeightedMessagePassing()
        self._cached_edge_index = None

    def reset_parameters(self):
        glorot(self.lin.weight)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, edge_index):
        if self._cached_edge_index is None:
            self._cached_edge_index = gcn_norm(  # yapf: disable
                edge_index, None, x.size(self.message_passing.node_dim),
                False, True, dtype=x.dtype)
        edge_index_normalized, edge_weight_normalized = self._cached_edge_index

        out = self.message_passing.propagate(edge_index_normalized, x=x,
                                             edge_weight=edge_weight_normalized)
        out = self.lin(out)
        return out


# The modules CRD, CLS, Net are from
# https://github.com/russellizadi/ssp/blob/master/experiments/gcn.py.

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


class Net(Module):
    def __init__(self, dataset, hidden_layers: int = 1, hidden_dim: int = 16):
        super(Net, self).__init__()
        in_features = dataset.num_features
        self.crds = ModuleList()
        for i in range(hidden_layers):
            self.crds.append(CRD(in_features, hidden_dim, 0.5))
            in_features = hidden_dim

        self.cls = CLS(in_features, dataset.num_classes)

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
    out = model(data)
    #loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    return loss, out

def train(model, optimizer, data, preconditioner=None, epoch=None):
    model.train()
    optimizer.zero_grad()

    if isinstance(optimizer, HessianFree):
        optimizer.step(lambda: model_forward(model), test_deterministic=True)
    else:
        loss = None
        if preconditioner is not None:
            with preconditioner.track_forward():
                out = model(data)
                label = out.max(1)[1]
                label[data.train_mask] = data.y[data.train_mask]
                label.requires_grad = False

                loss = F.nll_loss(out[data.train_mask], label[data.train_mask])
            with preconditioner.track_backward():
                loss.backward(retain_graph=True)

            update = epoch % 50 == 0
            if update:
                preconditioner.update_cov()
            preconditioner.step()
        else:
            out = model(data)
            label = out.max(1)[1]
            label[data.train_mask] = data.y[data.train_mask]

            loss = F.nll_loss(out[data.train_mask], label[data.train_mask])
            loss.backward(retain_graph=True)
        optimizer.step()


def evaluate(model, data):
    model.eval()

    with no_grad():
        logits = model(data)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{} loss'.format(key)] = loss
        outs['{} acc'.format(key)] = acc

    return outs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--baseline', choices=['adam', 'hessian', 'ggn'], default='adam')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--hidden_layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--kfac_damping', type=float, default=0.1)
    parser.add_argument('--hessianfree_damping', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--wandb', action='store_true', help='Track experiment')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    name = args.dataset
    dataset = Planetoid(join(dirname(__file__), '..', 'data', name), name)
    dataset.transform = T.NormalizeFeatures()
    data = dataset[0].to(device)

    epochs = args.epochs
    gamma = None
    runs = 1

    def train_model(params):
        enable_kfac = params["kfac"]
        model = Net(dataset, hidden_layers=args.hidden_layers, hidden_dim=args.hidden_channels)
        lr = args.lr
        weight_decay = args.weight_decay

        model.to(device).reset_parameters()
        if args.baseline in ["hessian", "ggn"] and not enable_kfac:
            optimizer = HessianFree(model.parameters(), verbose=False, curvature_opt=args.baseline, cg_max_iter=1000,
                                    damping=args.hessianfree_damping, adapt_damping=False)
        else:
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            #optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

        print(f"Optimizer: {optimizer.__class__} (specified {args.baseline})")
        preconditioner = None
        if enable_kfac:
            # damping corresponds to eps in PSGD.
            # cov_ema_decay corresponds to 1-alpha in PSGD.
            # PSGD does not implement momentum.
            preconditioner = KFAC(model, 0, args.kfac_damping, momentum=0.0, damping_adaptation_decay=0,
                                  cov_ema_decay=0.0, enable_pi_correction=False,
                                  damping_adaptation_interval=1, update_cov_manually=True)
            linear_blocks = sum(1 for block in preconditioner.blocks if isinstance(block, FullyConnectedFisherBlock))
            print(f"Preconditioning active on {linear_blocks} blocks.")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        epochs_tqdm = tqdm(range(1, epochs + 1))
        losses, val_accuracies, test_accuracies, train_accuracies = [], [], [], []
        for epoch in epochs_tqdm:
            train(model, optimizer, data, preconditioner, epoch=epoch-1)
            eval_info = evaluate(model, data)
            epochs_tqdm.set_description(
                f"Val loss:{eval_info['val loss']:.2f}, Train loss: {eval_info['train loss']:.2f}, Test acc: {eval_info['test acc']:.2f}")
            losses.append(eval_info['train loss'])
            val_accuracies.append(eval_info['val acc'] * 100)
            test_accuracies.append(eval_info['test acc'] * 100)
            train_accuracies.append(eval_info['train acc'] * 100)

        return losses, val_accuracies, test_accuracies, train_accuracies

    _, _, fig = plot_training(args.epochs, args.runs, train_model, [
        {"kfac": False}, {"kfac": True}
    ], [args.baseline, "KFAC"])

    fig.suptitle(f"GCN Training on {args.dataset}")
    fig.savefig(f"GCN-loss-{args.dataset}.svg")
