import argparse
import os.path as osp
import time

import numpy
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import MLP, GINConv, global_mean_pool, BatchNorm, global_add_pool
from torch_geometric.transforms import RandomLinkSplit

from torch_kfac import KFAC
from torch_kfac.layers import FullyConnectedFisherBlock
from tqdm.auto import tqdm



class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout,
                 aggregation):
        super().__init__()

        self.pooling = None
        if aggregation == "mean":
            self.pooling = global_mean_pool
        elif aggregation == "sum":
            self.pooling = global_add_pool

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # original:
            # mlp = MLP([in_channels, hidden_channels, hidden_channels])
            mlp = Sequential(Linear(in_channels, hidden_channels),
                             ReLU(),
                             Linear(hidden_channels, hidden_channels),
                             BatchNorm(hidden_channels),
                             Dropout(dropout)
                             )
            self.convs.append(GINConv(nn=mlp, train_eps=True))
            in_channels = hidden_channels

        # original:
        #self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
        #               norm=None, dropout=0.5)
        self.mlp = Sequential(Linear(hidden_channels, hidden_channels),
                              ReLU(),
                              Linear(hidden_channels, out_channels)
                              )

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = self.pooling(x, batch)
        return self.mlp(x)


def train(epoch, enable_kfac):
    model.train()

    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        if enable_kfac:
            with preconditioner.track_forward():
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                loss = F.cross_entropy(out, data.y)
            with preconditioner.track_backward():
                loss.backward()
        else:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
        if enable_kfac:
            if epoch % 10 == 0:
                preconditioner.update_cov()
            preconditioner.step()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PROTEINS')  # MUTAG
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--kfac_damping', type=float, default=1e-7)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    args = parser.parse_args()
    print(args)
    enable_kfac = args.kfac_damping is not None and args.kfac_damping != 0.0
    print(f"Enable kfac: {enable_kfac}")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS is currently slower than CPU due to missing int64 min/max ops
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
    dataset = TUDataset(path, name=args.dataset).shuffle()

    train_loader = DataLoader(dataset[:0.8], args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset[0.8:0.9], args.batch_size)
    test_loader = DataLoader(dataset[0.9:], args.batch_size)

    test_accuracies = []
    runs = 20
    for run in range(runs):
        model = GIN(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        preconditioner = KFAC(model, 0, args.kfac_damping, enable_pi_correction=False,
                              cov_ema_decay=0.85, momentum=0, update_cov_manually=True)

        if enable_kfac:
            linear_blocks = sum(1 for block in preconditioner.blocks if isinstance(block, FullyConnectedFisherBlock))
            print(f"Preconditioning active on {linear_blocks} blocks.")

        times = []
        best_test_acc = 0.
        best_val_acc = 0.
        epochs_tqdm = tqdm(range(args.epochs))
        for epoch in epochs_tqdm:
            start = time.time()
            loss = train(epoch, enable_kfac)
            train_acc = test(train_loader)
            val_acc = test(val_loader)
            test_acc = test(test_loader)
            epochs_tqdm.set_description(f"Loss: {loss:.3f}, Val acc: {val_acc*100:.2f}%, Test acc: {test_acc*100:.2f}%")

            times.append(time.time() - start)

            if val_acc > best_val_acc:
                best_test_acc = test_acc
                best_val_acc = val_acc
        test_accuracies.append(best_test_acc)
        print(f'Median time per epoch: {torch.tensor(times).median():.4f}s, Best test accuracy: {best_test_acc*100:.2f}%')

    print(f'Overall Test accuracy: {numpy.mean(test_accuracies)*100:.2f}%±{numpy.std(test_accuracies)*100:.2f}%')
