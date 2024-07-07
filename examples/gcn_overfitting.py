import time
from random import sample

from pandas import DataFrame
from torch import Tensor, no_grad
from torch.nn import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_random_edge

from examples.linear_block import LinearBlock
from planetoid import GCN
import torch
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import log
import argparse
import os.path as osp
import torch.nn as nn

from torch_kfac import KFAC
from torch_kfac.layers.fisher_block_factory import FisherBlockFactory


from torch_kfac.layers import FullyConnectedFisherBlock


class LinearBlockOverfit(FullyConnectedFisherBlock):
    """
    A linear block for linear layers of a graph neural network.
    The input and the output gradients are filtered by the training mask.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mask = None
        self.add_nodes = 0

    @no_grad()
    def backward_hook(self, module: Linear, grad_inp: Tensor, grad_out: Tensor) -> None:
        assert self.train_mask is not None
        new_mask = self.train_mask
        unselected_nodes = torch.arange(0, len(self.train_mask))[~self.train_mask.to("cpu")]
        random_nodes = sample(unselected_nodes.tolist(), self.add_nodes)
        new_mask[random_nodes] = True
        grad_out = (grad_out[0][new_mask],)
        super(LinearBlockOverfit, self).backward_hook(module, grad_inp, grad_out)
        new_mask[random_nodes] = False

    @no_grad()
    def forward_hook(self, module: Linear, input_data: Tensor, output_data: Tensor) -> None:
        assert self.train_mask is not None
        new_mask = self.train_mask
        unselected_nodes = torch.arange(0, len(self.train_mask))[~self.train_mask.to("cpu")]
        random_nodes = sample(unselected_nodes.tolist(), self.add_nodes)
        new_mask[random_nodes] = True

        input_data = (input_data[0][new_mask],)
        super(LinearBlockOverfit, self).forward_hook(module, input_data, output_data)
        new_mask[random_nodes] = False



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
parser.add_argument('--kfac_damping', type=float, default=0.1)
parser.add_argument('--cov_ema_decay', type=float, default=0.0)
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)

def init_trainig():
    model = GCN(dataset, hidden_dim=args.hidden_channels).to(device)

    optimizer = torch.optim.Adam([
        dict(params=model.crds.parameters(), weight_decay=5e-4),
        dict(params=model.cls.parameters(), weight_decay=0)
    ], lr=args.lr)  # Only perform weight-decay on first convolution.

    #factory = FisherBlockFactory([(nn.Linear, LinearBlock)])
    factory = None
    preconditioner = KFAC(model, 0.0, args.kfac_damping, momentum=0.0, damping_adaptation_decay=0.99,
                          cov_ema_decay=args.cov_ema_decay, enable_pi_correction=True, adapt_damping=True,
                          damping_adaptation_interval=5,
                          block_factory=factory)

    return model, optimizer, preconditioner


def train(model, optimizer, preconditioner, add_node_probability):
    model.train()
    optimizer.zero_grad()

    original_edge_index = data.edge_index.clone()

    edge_index, added_edges = add_random_edge(data.edge_index, p=add_node_probability)
    data.edge_index = edge_index
    added_count = added_edges.shape[1]
    # remove adjacency matrix cache.
    for conv in model.crds:
        conv._cached_edge_index = gcn_norm(  # yapf: disable
                data.edge_index, None, data.num_nodes,
                False, True, dtype=data.x.dtype)
        #conv._cached_edge_index[1][-added_count:] *= 0.01

    model.cls._cached_edge_index = gcn_norm(  # yapf: disable
                data.edge_index, None, data.num_nodes,
                False, True, dtype=data.x.dtype)
    #model.cls._cached_edge_index[1][-added_count:] *= 0.01


    with preconditioner.track_forward():
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    with preconditioner.track_backward():
        loss.backward()
    preconditioner.step(loss)
    optimizer.step()
    data.edge_index = original_edge_index
    return float(loss)


@torch.no_grad()
def test(model):
    model.eval()
    pred = model(data).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

results = []

for add_node_probability in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.25]:
    test_accs = []
    losses = []
    for _ in range(args.runs):
        model, optimizer, preconditioner = init_trainig()

        best_val_acc = test_acc = 0
        best_loss = 0
        times = []
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            loss = train(model, optimizer, preconditioner, add_node_probability)
            train_acc, val_acc, tmp_test_acc = test(model)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_loss = loss
                if tmp_test_acc > test_acc:
                    test_acc = tmp_test_acc
            log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
            times.append(time.time() - start)
        print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
        test_accs.append(test_acc)
        losses.append(best_loss)
    results.append([add_node_probability*100,
                    round(torch.mean(torch.tensor(test_accs)).item(), 3),
                    round(torch.std(torch.tensor(test_accs)).item(), 3),
                    round(torch.mean(torch.tensor(losses)).item(), 3),
                    round(torch.std(torch.tensor(losses)).item(), 3)])

df = DataFrame(results, columns=["add_nodes", "best_test_acc_mean", "best_test_acc_std", "best_loss_mean", "best_loss_std"])
df.to_csv(osp.join(osp.dirname(osp.realpath(__file__)), "add-nodes.csv"), index=False)
print(df)
