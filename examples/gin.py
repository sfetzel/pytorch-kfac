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
from torch_geometric.nn import MLP, GINConv, global_add_pool, BatchNorm

from torch_kfac import KFAC
from torch_kfac.layers import FullyConnectedFisherBlock




class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # original:
            # mlp = MLP([in_channels, hidden_channels, hidden_channels])
            mlp = Sequential(Linear(in_channels, hidden_channels),
                             BatchNorm(hidden_channels),
                             ReLU(),
                             Linear(hidden_channels, hidden_channels))
            self.convs.append(GINConv(nn=mlp, train_eps=True))
            in_channels = hidden_channels

        # original:
        #self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
        #               norm=None, dropout=0.5)
        self.mlp = Sequential(Linear(hidden_channels, hidden_channels),
                              ReLU(),
                              Dropout(0.5),
                              Linear(hidden_channels, out_channels),
                              Dropout(0.5))

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x)


def train(epoch):
    model.train()

    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        with preconditioner.track_forward():
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
        with preconditioner.track_backward():
            loss.backward()
        if args.kfac:
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
    parser.add_argument('--wandb', action='store_true', help='Track experiment')
    parser.add_argument('--kfac', action='store_true', default=False)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS is currently slower than CPU due to missing int64 min/max ops
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    init_wandb(
        name=f'GIN-{args.dataset}',
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        device=device,
    )

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
    dataset = TUDataset(path, name=args.dataset).shuffle()

    train_loader = DataLoader(dataset[:0.9], args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset[0.9:], args.batch_size)

    test_accuracies = []
    runs = 20
    for run in range(runs):
        model = GIN(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=args.num_layers,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        preconditioner = KFAC(model, 0, 1e-7, cov_ema_decay=0.85, momentum=0, update_cov_manually=True)

        if args.kfac:
            linear_blocks = sum(1 for block in preconditioner.blocks if isinstance(block, FullyConnectedFisherBlock))
            print(f"Preconditioning active on {linear_blocks} blocks.")

        times = []
        best_test_acc = 0.
        for epoch in range(args.epochs):
            start = time.time()
            loss = train(epoch)
            train_acc = test(train_loader)
            test_acc = test(test_loader)
            log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
            times.append(time.time() - start)
            best_test_acc = max(best_test_acc, test_acc)
        test_accuracies.append(best_test_acc)
        print(f'Median time per epoch: {torch.tensor(times).median():.4f}s, Best test accuracy: {best_test_acc*100:.2f}%')

    print(f'Test accuracy: {numpy.mean(test_accuracies)*100:.2f}%±{numpy.std(test_accuracies)*100:.2f}%')
