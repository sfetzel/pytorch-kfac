import os
from os.path import join, dirname

import torch
from torch import Tensor, no_grad
from torch.nn import Module, Linear, ModuleList
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm, GCNConv
from torch_geometric.nn.inits import glorot
import torch_geometric.transforms as T
from torch_sparse import SparseTensor, matmul
import torch.nn.functional as F

from torch_kfac import KFAC

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

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


class MyGcnConv(Module):
    def __init__(self, d_in, d_out, cached=True, bias=True):
        super(MyGcnConv, self).__init__()
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

    def forward(self, x, edge_index, mask=None):
        if self._cached_edge_index is None:
            self._cached_edge_index = gcn_norm(  # yapf: disable
                edge_index, None, x.size(self.message_passing.node_dim),
                False, True, dtype=x.dtype)
        edge_index_normalized, edge_weight_normalized = self._cached_edge_index
        out = x
        out = self.message_passing.propagate(edge_index_normalized, x=out,
                                             edge_weight=edge_weight_normalized)
        out = self.lin(out)
        return out

class CRD(Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = MyGcnConv(d_in, d_out, cached=True)
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
        #self.conv = GCNConv(d_in, d_out, cached=True)
        self.conv = MyGcnConv(d_in, d_out, cached=True)

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


def train(model, optimizer, data, preconditioner=None, lam=0.):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    label = out.max(1)[1]
    label[data.train_mask] = data.y[data.train_mask]
    label.requires_grad = False

    loss = F.nll_loss(out[data.train_mask], label[data.train_mask])
    loss += lam * F.nll_loss(out[~data.train_mask], label[~data.train_mask])
    loss.backward(retain_graph=True)
    if preconditioner:
        preconditioner.step(lam=lam)
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
    seed_everything()
    model = Net(dataset)
    lr = 0.01
    weight_decay = 0.0005
    data = dataset[0].to(device)
    #preconditioner = KFAC(model, lr, 1e-7)
    preconditioner = None

    torch.manual_seed(42)
    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = 200
    gamma = None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for epoch in range(1, epochs + 1):
        lam = (float(epoch) / float(epochs)) ** gamma if gamma is not None else 0.
        train(model, optimizer, data, preconditioner)
        eval_info = evaluate(model, data)

    print(f"Val loss: {eval_info['val loss']:.4f}, Test accuracy: {eval_info['test acc'] * 100}%")