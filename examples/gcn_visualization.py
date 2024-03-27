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
import logging
from torch_kfac import KFAC
from torch_kfac.layers import FullyConnectedFisherBlock

# Get torch_landscape from https://github.com/SamirMoustafa/visualization-loss-landscape-GNNs/.
from torch_landscape.directions import PcaDirections, LearnableDirections
from torch_landscape.landscape_linear import LinearLandscapeCalculator
from torch_landscape.trajectory import TrajectoryCalculator
from torch_landscape.utils import clone_parameters, reset_parameters, seed_everything
from torch_landscape.visualize import Plotly2dVisualization, VisualizationData
from torch_landscape.visualize_options import VisualizationOptions

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

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
    def __init__(self, dataset, n_layers: int = 1, hidden_dim: int = 16):
        super(Net, self).__init__()
        in_features = dataset.num_features
        self.crds = ModuleList()
        for i in range(n_layers):
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

preconditioning_start = 150

def train(model, optimizer, data, preconditioner=None, lam=0., epoch=None):
    model.train()
    optimizer.zero_grad()
    loss = None
    with preconditioner.track_forward():
        out = model(data)
        label = out.max(1)[1]
        label[data.train_mask] = data.y[data.train_mask]
        label.requires_grad = False

        loss = F.nll_loss(out[data.train_mask], label[data.train_mask])
        loss += lam * F.nll_loss(out[~data.train_mask], label[~data.train_mask])
    with preconditioner.track_backward():
        loss.backward(retain_graph=True)

    if epoch >= preconditioning_start:
        update = epoch % 50 == 0
        if update:
            preconditioner.update_cov()
        preconditioner.step()

    optimizer.step()
    return clone_parameters(model.parameters()), loss.item()


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

    epochs = 200
    gamma = None
    runs = 1

    model = Net(dataset, n_layers=1, hidden_dim=16)
    lr = 0.01
    weight_decay = 0.0005
    data = dataset[0].to(device)

    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    preconditioner = KFAC(model, 0, 0.01, momentum=0.0, damping_adaptation_decay=0,
                          cov_ema_decay=0.0, enable_pi_correction=True,
                          damping_adaptation_interval=1, update_cov_manually=True)
    linear_blocks = sum(1 for block in preconditioner.blocks if isinstance(block, FullyConnectedFisherBlock))
    print(f"Preconditioning active on {linear_blocks} blocks.")

    t_start = time.perf_counter()
    best_eval_info = None
    intermediate_parameters = []
    epochs_tqdm = tqdm(range(1, epochs + 1))
    parameters_with_loss = []
    for epoch in epochs_tqdm:
        parameters_with_loss.append(train(model, optimizer, data, preconditioner, epoch=epoch-1))
        eval_info = evaluate(model, data)
        epochs_tqdm.set_description(
            f"Val loss:{eval_info['val loss']:.2f}, Train loss: {eval_info['train loss']:.2f}")
        if best_eval_info is None or eval_info['val loss'] < best_eval_info['val loss']:
            best_eval_info = eval_info

    directions = LearnableDirections([*model.parameters()], parameters_with_loss, autoencoder_lr=0.001).calculate_directions()

    options = VisualizationOptions(num_points=15, use_log_z=True)
    trajectory = TrajectoryCalculator([*model.parameters()], directions).project_with_loss(parameters_with_loss)
    trajectory.set_range_to_fit_trajectory(options)
    landscape_calculator = LinearLandscapeCalculator(model.parameters(), directions, options=options, n_jobs=4)
    landscape = landscape_calculator.calculate_loss_surface_data_model(model, lambda: evaluate(model, data)["train loss"])
    file_path = "gcn_visualization"
    title = f"GCN Optimizer Trajectory"
    Plotly2dVisualization(options).plot(VisualizationData(landscape, trajectory), file_path, title, "pdf")
    Plotly2dVisualization(options).plot(VisualizationData(landscape, trajectory), file_path, title, "html")
