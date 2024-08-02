import gc
import os
import json
from pathlib import Path
from collections import defaultdict

import requests
import argparse
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold

from torch.optim import Adam
from torch import no_grad, cuda, long, device as torch_device, random as torch_random, load, save, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.nn import Module, CrossEntropyLoss, ModuleList
from torch_geometric.transforms import OneHotDegree
from torch_geometric.nn import global_add_pool

from torch_geometric.utils import degree
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from examples.gat_conv import GATConv
#from gin import GIN
from torch_kfac import KFAC
from torch_kfac.layers import FullyConnectedFisherBlock, PyGLinearBlock
from torch_kfac.layers.fisher_block_factory import FisherBlockFactory

import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool


class GIN(torch.nn.Module):

    #def __init__(self, dim_features, dim_target, config):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout,
                 aggregation):
        super(GIN, self).__init__()

        self.dropout = dropout
        self.embeddings_dim = [hidden_channels] * num_layers
        self.num_layers = num_layers
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []

        train_eps = True

        self.pooling = None
        if aggregation == "mean":
            self.pooling = global_mean_pool
        elif aggregation == "sum":
            self.pooling = global_add_pool

        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.first_h = Sequential(Linear(in_channels, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                    Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.linears.append(Linear(out_emb_dim, out_channels))
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                      Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))  # Eq. 4.2

                self.linears.append(Linear(out_emb_dim, out_channels))

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input

    def forward(self, x, edge_index, batch):
        #x, edge_index, batch = data.x, data.edge_index, data.batch

        out = 0

        for layer in range(self.num_layers):
            if layer == 0:
                x = self.first_h(x)
                out += F.dropout(self.pooling(self.linears[layer](x), batch), p=self.dropout)
            else:
                # Layer l ("convolution" layer)
                x = self.convs[layer-1](x, edge_index)
                out += F.dropout(self.linears[layer](self.pooling(x, batch)), p=self.dropout, training=self.training)

        return out



class GAT(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout, hidden_layers):
        super().__init__()
        self.convs = ModuleList()
        for _ in range(hidden_layers):
            self.convs.append(GATConv(in_channels, hidden_channels, heads, dropout=dropout))
            in_channels = hidden_channels * heads
        self.conv_last = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=dropout)

        self.pooling = None
        if aggregation == "mean":
            self.pooling = global_mean_pool
        elif aggregation == "sum":
            self.pooling = global_add_pool

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        x = self.conv_last(x, edge_index)
        x = self.pooling(x, batch)
        return x

class Patience:
    """
    Implement common "patience" for early stopping.
    """
    def __init__(self, patience=20, use_loss=True, save_path=None):
        self.local_val_optimum = float("inf") if use_loss else -float("inf")
        self.use_loss = use_loss
        self.patience = patience
        self.best_epoch = -1
        self.counter = -1
        self.val_loss, self.val_acc = None, None
        self.save_path = save_path

    def stop(self, epoch, val_loss, val_acc=None, model=None):
        if self.use_loss:
            if val_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_loss
                self.best_epoch = epoch
                self.val_loss, self.val_acc = val_loss, val_acc
                self.model = model
                if all([model is not None, self.save_path is not None]):
                    save({"epoch": epoch + 1, "state_dict": model.state_dict()}, self.save_path)
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
        else:
            if val_acc >= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_acc
                self.best_epoch = epoch
                self.val_loss, self.val_acc = val_loss, val_acc
                self.model = model
                if all([model is not None, self.save_path is not None]):
                    if not os.path.exists(os.path.dirname(self.save_path)):
                        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                    save({"epoch": epoch + 1, "state_dict": model.state_dict()}, self.save_path)
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience


def train(model, data, y, criterion, optimizer, scheduler, preconditioner, enable_preconditioner, epoch):
    optimizer.zero_grad()
    if enable_preconditioner and epoch % 50 == 0:
        with preconditioner.track_forward():
            output = model(data.x, data.edge_index, data.batch)
            loss_train = criterion(output, y.to(output.device))
        with preconditioner.track_backward():
            loss_train.backward()
    else:
        output = model(data.x, data.edge_index, data.batch)
        loss_train = criterion(output, y.to(output.device))
        loss_train.backward()

    if enable_preconditioner:
        try:
            if epoch % 50 == 0:
               preconditioner.update_cov()

            preconditioner.step(loss_train)
        except:
            print("Problem in KFAC preconditioner! Continuing without")

    optimizer.step()
    scheduler.step()
    return output, loss_train


@no_grad()
def eval(model, data, y, criterion):
    output = model(data.x, data.edge_index, data.batch)
    loss_test = criterion(output, y.to(output.device))
    return output, loss_test


def train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, preconditioner,
                             enable_preconditioner, epoch):
    train_loss = 0
    train_correct = 0
    model.train()
    for idx, data in enumerate(train_loader):
        data = data.to(device)
        output, loss = train(model, data, data.y, criterion, optimizer, scheduler, preconditioner, enable_preconditioner, epoch)
        train_loss += loss.item() * data.num_graphs
        prediction = output.max(1)[1].type_as(data.y)
        train_correct += prediction.eq(data.y.double()).sum().item()

    train_acc = train_correct / len(train_loader.dataset)
    train_loss = train_loss / len(train_loader.dataset)

    val_loss = 0
    val_correct = 0
    model.eval()
    for idx, data in enumerate(val_loader):
        data = data.to(device)
        output, loss = eval(model, data, data.y, criterion)
        val_loss += loss.item() * data.num_graphs
        prediction = output.max(1)[1].type_as(data.y)
        val_correct += prediction.eq(data.y.double()).sum().item()

    val_acc = val_correct / len(val_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)

    return train_acc * 100, train_loss, val_acc * 100, val_loss


def validate_batch_size(length, batch_size):
    return length % batch_size == 1


def get_train_val_test_loaders(dataset, train_index, val_index, test_index):
    train_set = [dataset[i] for i in train_index]
    val_set = [dataset[i] for i in val_index]
    test_set = [dataset[i] for i in test_index]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=validate_batch_size(len(train_set), args.batch_size))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def read_json_from_url(dataset_name):
    url_dict = {"DD": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/CHEMICAL/DD_splits.json",
                "ENZYMES": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/CHEMICAL/ENZYMES_splits.json",
                "NCI1": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/CHEMICAL/NCI1_splits.json",
                "PROTEINS": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/CHEMICAL/PROTEINS_full_splits.json",
                "IMDB-BINARY": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/COLLABORATIVE_DEGREE/IMDB-BINARY_splits.json",
                "IMDB-MULTI": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/COLLABORATIVE_DEGREE/IMDB-MULTI_splits.json",
                "COLLAB": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/COLLABORATIVE_DEGREE/COLLAB_splits.json",
                "REDDIT-BINARY": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/COLLABORATIVE_DEGREE/REDDIT-BINARY_splits.json",
                "REDDIT-MULTI-5K": "https://raw.githubusercontent.com/diningphil/gnn-comparison/master/data_splits/COLLABORATIVE_DEGREE/REDDIT-MULTI-5K_splits.json",
                }
    try:
        response = requests.get(url_dict[dataset_name])
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        return json.loads(response.text)
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    except KeyError:
        print(f"No splits available for {dataset_name}")
        return None


def compute_max_degree(dataset):
    max_degree = 0
    degrees = []
    for data in dataset:
        degrees += [degree(data.edge_index[0], dtype=long)]
        max_degree = max(max_degree, degrees[-1].max().item())
    return max_degree


def get_model(model_str: str, args, dataset, hidden_channels, num_layers, dropout, aggregation) -> Module:
    if model_str == "GIN":
        return GIN(
            in_channels=dataset.num_features,
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=num_layers,
            dropout=dropout,
            aggregation=aggregation
        )
    elif model_str == "GAT":
        return GAT(in_channels=dataset.num_features,
                   hidden_channels=hidden_channels,
                   out_channels=dataset.num_classes,
                   heads=args.heads,
                   dropout=dropout,
                   hidden_layers=num_layers-1,
                   aggregation=aggregation)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ENZYMES") #REDDIT-BINARY
    # if dataset has a lot of features, then use a larger embedding size; e.g. 32, 64, 128 for cora
    # look at paper (https://arxiv.org/pdf/1912.09893.pdf) table 6 for suggestions
    parser.add_argument("--dim_embeddings", type=int, nargs="+", default=[32, 64]) # 32, 64;
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.01]) # 0.01, 1e-3; 1e-3 just increases std
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 3, 5]) # 1, 2, 3, 4; use even or odd numbers

    # if there are 4 choices for damping, more tendency towards preconditioning?
    # for proof of concept: without reporting accuracies
    parser.add_argument("--kfac_damping", nargs="+", default=[0.1, None])# None, 0.01, 1e-7
    parser.add_argument("--weight_decay", type=float, nargs="+", default=[0.0005, 0.0])
    parser.add_argument("--aggregation", type=str, nargs="+", default=["mean", "sum"])
    parser.add_argument("--dropout", type=int, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_selection_metric", type=str, default="accuracy", choices=["accuracy", "loss"])
    parser.add_argument("--device", type=str, default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--heads", type=int, default=8, help='Heads for GAT')
    parser.add_argument('--model', type=str, required=True, choices=["GIN", "GAT"])
    args = parser.parse_args()

    device = torch_device(args.device) if all([args.device[:4] == "cuda", cuda.is_available()]) else torch_device("cpu")
    print(device)
    root = Path(__file__).resolve().parent.parent.joinpath("data")

    dataset = TUDataset(root=root, name=args.dataset_name)

    args.kfac_damping = [float(damping) if (damping != "None" and damping is not None) else None for damping in args.kfac_damping]

    dataset_args = {"use_node_attr": False, "transform": None}
    # Update `dataset_args` based on `dataset_name`
    if "ENZYMES" in args.dataset_name:
        dataset_args.update(use_node_attr=True)
    elif "PROTEINS" in args.dataset_name:
        dataset_args.update(use_node_attr=True)
    elif "IMDB-BINARY" in args.dataset_name:
        max_degree = compute_max_degree(dataset)
        dataset_args.update(transform=OneHotDegree(max_degree=max_degree))
    elif "IMDB-MULTI" in args.dataset_name:
        max_degree = compute_max_degree(dataset)
        dataset_args.update(transform=OneHotDegree(max_degree=max_degree))
    elif "COLLAB" in args.dataset_name:
        max_degree = compute_max_degree(dataset)
        dataset_args.update(transform=OneHotDegree(max_degree=max_degree))
    elif "REDDIT-BINARY" in args.dataset_name:
        max_degree = compute_max_degree(dataset)
        dataset_args.update(transform=OneHotDegree(max_degree=max_degree))
    elif "REDDIT-MULTI-5K" in args.dataset_name:
        max_degree = compute_max_degree(dataset)
        dataset_args.update(transform=OneHotDegree(max_degree=max_degree))

    dataset = TUDataset(root=root, name=args.dataset_name, **dataset_args)

    y = [d.y.item() for d in dataset]
    features_dim = dataset[0].x.shape[1]
    n_classes = len(np.unique(y))
    criterion = CrossEntropyLoss()
    splits = read_json_from_url(args.dataset_name)

    if splits is None:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
        splits = [{"model_selection": [{"train": train_index.tolist(),
                                        "validation": test_index.tolist()}],
                   "test": test_index.tolist()}
                  for it, (train_index, test_index) in enumerate(skf.split(np.zeros(len(y)), y))]

    all_accuracies_folds = []
    mean_accuracies_folds = []
    best_config_across_folds = [] # best hyperparameters; e.g with best LR, best batch size

    np.random.seed(10)
    torch_random.manual_seed(10)
    preconditioner_args = {"momentum": 0, "update_cov_manually":True}

    for it in range(10):
        # it corresponds to the fold.
        print("-" * 30 + f"ITERATION {str(it + 1)}" + "-" * 30)
        train_index = splits[it]["model_selection"][0]["train"]
        val_index = splits[it]["model_selection"][0]["validation"]
        test_index = splits[it]["test"]

        loop_counter = 1
        result_dict = defaultdict(list)
        best_acc_across_folds = -float(np.inf)
        best_loss_across_folds = float(np.inf)

        n_params = len(args.dim_embeddings) * len(args.lrs) * len(args.kfac_damping) * len(args.layers) * len(args.dropout) * len(args.weight_decay) * len(args.aggregation)
        model_selection_epochs = args.epochs
        if n_params == 1:
            print(f"Only one configuration to search for, skipping model selection (train for one epoch)")
            model_selection_epochs = 1
        for lr in args.lrs:
            for dim_embedding in args.dim_embeddings:
                for kfac_damping in args.kfac_damping:
                    for layers in args.layers:
                        for dropout in args.dropout:
                            for weight_decay in args.weight_decay:
                                for aggregation in args.aggregation:
                                    ################################
                                    #       MODEL SELECTION       #
                                    ###############################
                                    train_loader, val_loader, test_loader = get_train_val_test_loaders(dataset,
                                                                                                       train_index,
                                                                                                       val_index,
                                                                                                       test_index,
                                                                                                       )

                                    early_stopper = Patience(patience=args.patience, use_loss=False)
                                    # dont forget to add hyperparameters here.
                                    params = {"dim_embedding": dim_embedding, "lr": lr, "dropout": dropout,
                                              "layers": layers, "kfac_damping": kfac_damping,
                                              "weight_decay": weight_decay, "aggregation": aggregation}

                                    best_val_loss, best_val_acc = float("inf"), 0
                                    epoch, val_loss, val_acc = 0, float("inf"), 0

                                    model = get_model(args.model, args, dataset, dim_embedding, layers, dropout, aggregation)

                                    model = model.to(device)

                                    print(f"Model # Parameters {sum([p.numel() for p in model.parameters()])}")
                                    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                                    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.5)
                                    # we can use the regular preconditioning blocks, because
                                    # only the training data is used for the loss.
                                    factory = FisherBlockFactory()
                                    preconditioner = KFAC(model, 0, kfac_damping if kfac_damping is not None else 0, **preconditioner_args,
                                                          block_factory=factory)
                                    if kfac_damping is not None:
                                        linear_blocks = sum(1 for block in preconditioner.blocks if isinstance(block, FullyConnectedFisherBlock))
                                        print(f"Preconditioning active on {linear_blocks} blocks.")
                                    else:
                                        print("Preconditioning inactive.")

                                    pbar_train = tqdm(range(model_selection_epochs), desc="Epoch 0 Loss 0")
                                    for epoch in pbar_train:
                                        train_acc, train_loss, val_acc, val_loss = train_and_validate_model(model, train_loader,
                                                                                                            val_loader, criterion,
                                                                                                            optimizer, scheduler,
                                                                                                            device,
                                                                                                            preconditioner,
                                                                                                            kfac_damping is not None, epoch
                                                                                                            )

                                        if early_stopper.stop(epoch, val_loss, val_acc):
                                            break

                                        best_acc_across_folds = early_stopper.val_acc if early_stopper.val_acc > best_acc_across_folds else best_acc_across_folds
                                        best_loss_across_folds = early_stopper.val_loss if early_stopper.val_loss < best_loss_across_folds else best_loss_across_folds

                                        pbar_train.set_description(f"MS {loop_counter}/{n_params} Epoch {epoch + 1} Val loss {val_loss:0.2f} Val acc {val_acc:0.1f} Best Val Loss {early_stopper.val_loss:0.2f} Best Val Acc {early_stopper.val_acc:0.1f} Best Fold Val Acc  {best_acc_across_folds:0.1f} Best Fold Val Loss {best_loss_across_folds:0.2f}")

                                    best_val_loss, best_val_acc = early_stopper.val_loss, early_stopper.val_acc

                                    result_dict["config"].append(params)
                                    result_dict["best_val_acc"].append(best_val_acc)
                                    result_dict["best_val_loss"].append(best_val_loss)
                                    print(f"MS {loop_counter}/{n_params} Epoch {epoch + 1} Best Epoch {early_stopper.best_epoch} Val acc {val_acc:0.1f} Best Val Acc {best_val_acc:0.2f} Best Fold Val Acc  {best_acc_across_folds:0.2f} Best Fold Val Loss {best_loss_across_folds:0.2f}")

                                    test_count = 0
                                    test_correct = 0
                                    for idx, data in enumerate(test_loader):
                                        data = data.to(device)
                                        output, loss = eval(model, data, data.y, criterion)
                                        test_count += output.size(0)
                                        prediction = output.max(1)[1].type_as(data.y)
                                        test_correct += prediction.eq(data.y.double()).sum().item()

                                    test_accuracy = (test_correct / test_count) * 100
                                    print(f"Val acc {best_val_acc:0.2f}, Test acc {test_accuracy:0.2f}, f{params}")

                                    loop_counter += 1
                                    gc.collect()
                                    torch.cuda.empty_cache()

        # Free memory after model selection
        del model
        del optimizer
        del scheduler
        ################################
        #       MODEL ASSESSMENT      #
        ###############################
        # train with best configuration here.
        if args.model_selection_metric == "accuracy":
            best_i = np.argmax(result_dict["best_val_acc"])
        elif args.model_selection_metric == "loss":
            best_i = np.argmin(result_dict["best_val_loss"])
        best_config = result_dict["config"][best_i]
        best_val_acc = result_dict["best_val_acc"][best_i]
        print(best_config)
        print(f"Winner of Model Selection | hidden dim: {best_config['dim_embedding']} | lr {best_config['lr']}")
        print(f"Winner Best Val Accuracy {result_dict['best_val_acc'][best_i]:0.2f}")
        loop_counter = 1
        test_accuracies = []
        for _ in range(args.runs):
            # use best_config here!
            model = get_model(args.model, args, dataset, best_config["dim_embedding"], best_config["layers"], best_config["dropout"],
                              best_config["aggregation"])

            model = model.to(device)
            optimizer = Adam(model.parameters(), lr=best_config["lr"], weight_decay=best_config["weight_decay"])
            scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.5)
            kfac_damping = best_config["kfac_damping"]
            preconditioner = KFAC(model, 0, kfac_damping if kfac_damping is not None else 0, **preconditioner_args)

            save_path = os.path.join("models", args.dataset_name, "model_best" + ".pth.tar")
            early_stopper = Patience(patience=args.patience, use_loss=False, save_path=save_path)

            pbar_train = tqdm(range(args.epochs), desc="Epoch 0 Loss 0")

            for epoch in pbar_train:
                train_acc, train_loss, val_acc, val_loss = train_and_validate_model(model, train_loader, val_loader,
                                                                                    criterion, optimizer, scheduler,
                                                                                    device, preconditioner,
                                                                                    kfac_damping is not None, epoch)

                if early_stopper.stop(epoch, val_loss, val_acc, model=model):
                    break

                pbar_train.set_description(f"Test {loop_counter}/{args.runs} Epoch {epoch + 1} Val acc {val_acc:0.1f} Best Epoch {early_stopper.best_epoch} Best Val Acc {early_stopper.val_acc:0.2f}")

            checkpoint = load(save_path)
            epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])

            test_loss = 0
            test_count = 0
            test_correct = 0

            model.eval()

            for idx, data in enumerate(test_loader):
                data = data.to(device)
                output, loss = eval(model, data, data.y, criterion)
                test_loss += loss.item() * output.size(0)
                test_count += output.size(0)
                prediction = output.max(1)[1].type_as(data.y)
                test_correct += prediction.eq(data.y.double()).sum().item()

            test_accuracy = (test_correct / test_count) * 100
            print(f"Test {loop_counter}/{args.runs} "
                  f"Epoch {epoch + 1} "
                  f"Val loss {val_loss:0.2f} "
                  f"Val acc {val_acc:0.1f} "
                  f"Best Val Loss {early_stopper.val_loss:0.2f} "
                  f"Best Val Acc {early_stopper.val_acc:0.2f} "
                  f"Test acc {test_accuracy:0.2f}")

            loop_counter += 1
            test_accuracies.append(test_accuracy)

        all_accuracies_folds.append(test_accuracies)
        mean_accuracies_folds.append(np.mean(test_accuracies))
        best_config_across_folds.append(best_config)
        print(f"Test acc mean {mean_accuracies_folds[-1]:.3f}")

        print(f"Cross-val iter:{it + 1} | Current average test accuracy across folds {np.mean(mean_accuracies_folds):.5f}")
        print("\n")

    result_dict = {"all_test_accuracies": all_accuracies_folds,
                   "mean_test_accuracies": mean_accuracies_folds,
                   "best_params": best_config_across_folds}

    print(f"AVERAGE TEST ACC ACROSS FOLDS {np.mean(mean_accuracies_folds):.5f}")
    print(f"STD ACROSS FOLDS {np.std(mean_accuracies_folds)}")
