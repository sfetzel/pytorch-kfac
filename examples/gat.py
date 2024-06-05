import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from matplotlib import pyplot
from torch_geometric.datasets import Planetoid, OGB_MAG
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import BatchNorm

from gat_conv import GATConv
#from torch_geometric.nn.conv import GATConv
from tqdm.auto import tqdm
from planetoid import seed_everything
from torch_kfac import KFAC
from torch_kfac.layers import FullyConnectedFisherBlock

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
    parser.add_argument('--dataset', type=str, default='PubMed')
    parser.add_argument('--hidden_channels', type=int, default=8)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--kfac_damping', type=float, default=None, help='Set to none to disable preconditioning')
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--wandb', action='store_true', help='Track experiment')
    args = parser.parse_args()
    enable_preconditioning = args.kfac_damping is not None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_wandb(name=f'GAT-{args.dataset}', heads=args.heads, epochs=args.epochs,
               hidden_channels=args.hidden_channels, lr=args.lr, device=device)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    #path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'OgbMag')
    #dataset = OGB_MAG(path)
    data = dataset[0].to(device)

    #seed_everything()
    model = GAT(dataset.num_features, args.hidden_channels, dataset.num_classes,
                args.heads, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    preconditioner = KFAC(model, 0, args.kfac_damping if enable_preconditioning else 0, cov_ema_decay=0.0,
                          enable_pi_correction=False, update_cov_manually=False, momentum=0, damping_adaptation_decay=0,
                          damping_adaptation_interval=1)
    if enable_preconditioning:
        linear_blocks = sum(1 for block in preconditioner.blocks if isinstance(block, FullyConnectedFisherBlock))
        print(f"Preconditioning active on {linear_blocks} blocks.")

def train(enable_preconditioning):
    model.train()
    optimizer.zero_grad()
    if enable_preconditioning:
        with preconditioner.track_forward():
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        with preconditioner.track_backward():
            loss.backward()
    else:
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

    if enable_preconditioning:
        preconditioner.step()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


if __name__ == "__main__":
    times = []
    best_val_acc = final_test_acc = 0
    best_loss = 0
    epochs = range(1, args.epochs + 1)
    progress_bar = tqdm(epochs)
    losses = []
    val_accuracies = []
    test_accuracies = []
    train_accuracies = []
    for epoch in progress_bar:
        start = time.time()
        loss = train(enable_preconditioning)
        train_acc, val_acc, tmp_test_acc = test()
        losses.append(loss)
        val_accuracies.append(val_acc * 100)
        train_accuracies.append(train_acc * 100)
        test_accuracies.append(tmp_test_acc * 100)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            best_loss = loss
        progress_bar.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}, Train accuracy: {train_acc*100:.2f}%, Validation accuracy: {val_acc*100:.2f}%, Test accuracy: {test_acc*100:.2f}")
        times.append(time.time() - start)

    pyplot.title("GAT training - Loss")
    pyplot.plot(epochs, losses)
    pyplot.xlabel("epoch")
    pyplot.grid()
    pyplot.savefig("GAT-loss.svg")
    pyplot.clf()
    pyplot.title("GAT training - Accuracies")
    pyplot.plot(epochs, val_accuracies, label="Validation")
    pyplot.plot(epochs, train_accuracies, label="Training")
    pyplot.plot(epochs, test_accuracies, label="Test")
    pyplot.legend()
    pyplot.grid()
    pyplot.xlabel("epoch")
    pyplot.savefig("GAT-acc.svg")
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s, Best validation accuracy: {best_val_acc*100:.2f}% with test accuracy: {test_acc*100:.2f}%, loss: {best_loss:.4f}")
