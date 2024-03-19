import os
import time
from os.path import join, dirname
import torch
from torch import Tensor, no_grad, tensor
from torch.nn import Module, Linear, ModuleList
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot
import torch_geometric.transforms as T
from torch_sparse import SparseTensor, matmul
import torch.nn.functional as F
from tqdm.auto import tqdm

from torch_kfac import KFAC
from torch_kfac.layers import FullyConnectedFisherBlock

dev_mode = True


def seed_everything():
    """
    Sets the seed of torch, torch cuda, torch geometric and native random library.

    :param seed: (Optional) the seed to use.
    """
    seed = 42

    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import numpy as np

    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from torch_geometric import seed_everything

    seed_everything(seed)

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
        if dev_mode:
            self.lin.weight.data.fill_(0.1)
            self.bias.data.fill_(0.1)
        else:
            glorot(self.weight)
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
        if dev_mode:
            self.conv.lin.weight.data.fill_(0.1)
            self.conv.bias.data.fill_(0.1)
        else:
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
        if dev_mode:
            self.conv.lin.weight.data.fill_(0.1)
            self.conv.bias.data.fill_(0.1)
        else:
            self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x


class Net(Module):
    def __init__(self, dataset, n_layers: int = 1):
        super(Net, self).__init__()
        in_features = dataset.num_features
        self.crds = ModuleList()
        for i in range(n_layers):
            self.crds.append(CRD(in_features, 16, 0.5))
            in_features = 16

        self.cls = CLS(16, dataset.num_classes)

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

def train(model, optimizer, data, preconditioner=None, lam=0.):
    model.train()
    optimizer.zero_grad()
    with preconditioner.track_forward():
        out = model(data)
        label = out.max(1)[1]
        label[data.train_mask] = data.y[data.train_mask]
        label.requires_grad = False

        loss = F.nll_loss(out[data.train_mask], label[data.train_mask])
        loss += lam * F.nll_loss(out[~data.train_mask], label[~data.train_mask])
    with preconditioner.track_backward():
        loss.backward(retain_graph=True)
    if preconditioner:
        preconditioner.step()
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    name = "Cora"
    dataset = Planetoid(join(dirname(__file__), '..', 'data', name), name, split="public")
    dataset.transform = T.NormalizeFeatures()

    #seed_everything()
    epochs = 200
    gamma = None
    runs = 20

    val_losses, accuracies, durations = [], [], []
    for run in range(1, runs + 1):
        model = Net(dataset)
        lr = 0.01
        weight_decay = 0.0005
        data = dataset[0].to(device)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # damping corresponds to eps in PSGD.
        # cov_ema_decay corresponds to 1-alpha in PSGD.
        # PSGD does not implement momentum.
        preconditioner = KFAC(model, lr, 0.01, cov_ema_decay=0.0, momentum=0)
        linear_blocks = sum(1 for block in preconditioner.blocks if isinstance(block, FullyConnectedFisherBlock))
        print(f"Preconditioning active on {linear_blocks} blocks.")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        best_eval_info = None
        epochs_tqdm = tqdm(range(1, epochs + 1))
        for epoch in epochs_tqdm:
            lam = (float(epoch) / float(epochs)) ** gamma if gamma is not None else 0.
            train(model, optimizer, data, preconditioner)
            eval_info = evaluate(model, data)
            epochs_tqdm.set_description(f"Val loss:{eval_info['val loss']:.2f}")
            if best_eval_info is None or eval_info['val loss'] < best_eval_info['val loss']:
                best_eval_info = eval_info

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_eval_info['val loss'])
        accuracies.append(best_eval_info['test acc'])
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accuracies), tensor(durations)
    print('Val Loss: {:.4f}, Test Accuracy: {:.2f} ± {:.2f}, Duration: {:.3f} \n'.
          format(loss.mean().item(),
                 100*acc.mean().item(),
                 100*acc.std().item(),
                 duration.mean().item()))