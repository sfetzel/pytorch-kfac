import os
import json
from pathlib import Path
from collections import defaultdict

import requests
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from torch.optim import Adam
from torch import no_grad, cuda, long, device as torch_device, random as torch_random, load, save
from torch.optim.lr_scheduler import StepLR
from torch.nn import Module, CrossEntropyLoss
from torch_geometric.transforms import OneHotDegree


from torch_geometric.nn import global_max_pool
from torch_geometric.utils import degree
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from gin import GIN

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


def train(model, data, y, criterion, optimizer, scheduler):
    output = model(data.x, data.edge_index, data.batch)
    loss_train = criterion(output, y.to(output.device))
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    scheduler.step()
    return output, loss_train


@no_grad()
def eval(model, data, y, criterion):
    output = model(data.x, data.edge_index, data.batch)
    loss_test = criterion(output, y.to(output.device))
    return output, loss_test


def train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device):
    train_loss = 0
    train_correct = 0
    model.train()
    for idx, data in enumerate(train_loader):
        data = data.to(device)
        output, loss = train(model, data, data.y, criterion, optimizer, scheduler)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="REDDIT-BINARY") #REDDIT-BINARY
    parser.add_argument("--dim_embeddings", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.01,1e-3])
    #parser.add_argument("--kfac_damping", type=float, nargs="+", default=[0, 0.01, 1e-8])

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--runs", type=int, default=3)
    # todo: add argument for kfac optimizer (bool), # of layers, ..
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_bits", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch_device("cuda") if all([args.device == "cuda", cuda.is_available()]) else torch_device("cpu")

    root = Path(__file__).resolve().parent.parent.joinpath("data")

    dataset = TUDataset(root=root, name=args.dataset_name)

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
        # todo add new hyperparameters: damping, could also add dropout, number of layers, ..
        n_params = len(args.dim_embeddings) * len(args.lrs)
        model_selection_epochs = args.epochs
        if n_params == 1:
            print(f"Only one configuration to search for, skipping model selection (train for one epoch)")
            model_selection_epochs = 1
        for lr in args.lrs:
            for dim_embedding in args.dim_embeddings:

                ################################
                #       MODEL SELECTION       #
                ###############################
                train_loader, val_loader, test_loader = get_train_val_test_loaders(dataset,
                                                                                   train_index,
                                                                                   val_index,
                                                                                   test_index,
                                                                                   )

                early_stopper = Patience(patience=args.patience, use_loss=False)
                params = {"dim_embedding": dim_embedding, "lr": lr} # todo dont forget to add hyperparameters here.

                best_val_loss, best_val_acc = float("inf"), 0
                epoch, val_loss, val_acc = 0, float("inf"), 0

                # todo add num layers to args.
                model = GIN(in_channels=features_dim, num_layers=1,
                            hidden_channels=dim_embedding,
                            out_channels=n_classes)

                model = model.to(device)

                print(f"Model # Parameters {sum([p.numel() for p in model.parameters()])}")
                optimizer = Adam(model.parameters(), lr=lr)
                scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.5)

                pbar_train = tqdm(range(model_selection_epochs), desc="Epoch 0 Loss 0")
                for epoch in pbar_train:
                    train_acc, train_loss, val_acc, val_loss = train_and_validate_model(model, train_loader,
                                                                                        val_loader, criterion,
                                                                                        optimizer, scheduler,
                                                                                        device,
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

                loop_counter += 1

        # Free memory after model selection
        del model
        del optimizer
        del scheduler
        ################################
        #       MODEL ASSESSMENT      #
        ###############################
        # train with best configuration here.

        best_i = np.argmax(result_dict["best_val_acc"])
        best_config = result_dict["config"][best_i]
        best_val_acc = result_dict["best_val_acc"][best_i]
        print(f"Winner of Model Selection | hidden dim: {best_config['dim_embedding']} | lr {best_config['lr']}")
        print(f"Winner Best Val Accuracy {result_dict['best_val_acc'][best_i]:0.2f}")

        loop_counter = 1
        test_accuracies = []
        for _ in range(args.runs):
            # use best_config here!
            model = GIN(in_channels=features_dim, num_layers=1,
                                        hidden_channels=dim_embedding,
                                        out_channels=n_classes)

            model = model.to(device)
            optimizer = Adam(model.parameters(), lr=best_config["lr"])
            scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.5)

            save_path = os.path.join("models", args.dataset_name, "model_best" + ".pth.tar")
            early_stopper = Patience(patience=args.patience, use_loss=False, save_path=save_path)

            pbar_train = tqdm(range(args.epochs), desc="Epoch 0 Loss 0")

            for epoch in pbar_train:
                train_acc, train_loss, val_acc, val_loss = train_and_validate_model(model, train_loader, val_loader,
                                                                                    criterion, optimizer, scheduler,
                                                                                    device)

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