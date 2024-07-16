import argparse
import gc
import time

import torch
from matplotlib import pyplot
from torch import tensor, mean, std, argmax

from os.path import exists, splitext

from torch_operation_counter import OperationsCounterMode


def find_filename(filename, filename_ext=None):

    if filename_ext is None:
        filename, filename_ext = splitext(filename)
    i = 1
    while exists(f"{filename}-{i}.{filename_ext}"):
        i += 1

    return f"{filename}-{i}"

def set_fontsize(ax, font_size: float):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)


def plot_training(epoch_count, runs, training_fun, parameter_groups: list, captions: list,
                  loss_range=None, legend_loc='lower right'):
    times = []
    best_val_acc = final_test_acc = 0
    best_loss = 0
    epochs = range(1, epoch_count + 1)
    fontsize = 25
    fig, ax1 = pyplot.subplots(figsize=(15, 10))

    ax2 = ax1.twinx()
    set_fontsize(ax1, fontsize)
    set_fontsize(ax2, fontsize)

    linestyles = ["-", "--", "-."]
    group_results = {}
    for index, (parameter_group, caption) in enumerate(zip(parameter_groups, captions)):
        losses_van, val_accuracies_van, test_accuracies_van, train_accuracies_van = [], [], [], []

        cloned_args = parameter_group.copy()
        # only use one epoch to determine the FLOPs.
        cloned_args["epochs"] = 1
        # need to disable add_self_loops as this will use
        # a custom tensor type which is not supported by
        # torch-operation-counter.
        cloned_args["add_self_loops"] = False
        flops_per_epoch = None
        try:
            with OperationsCounterMode() as counter:
                training_fun(cloned_args)
                flops_per_epoch = counter.total_operations
        except Exception as e:
            print(f"Error when calculating FLOPs: {e}")
        best_loss, best_val_acc, best_test_acc, best_train_acc, best_epochs = [], [], [], [], []
        times = []
        for i in range(runs):
            start_time = time.perf_counter()
            losses, val_accuracies, test_accuracies, train_accuracies = training_fun(parameter_group)
            stop_time = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()
            losses_van.append(losses)
            times.append(stop_time - start_time)
            val_accuracies_van.append(val_accuracies)
            test_accuracies_van.append(test_accuracies)
            train_accuracies_van.append(train_accuracies)

            best_epoch = argmax(tensor(val_accuracies))
            best_loss.append(losses[best_epoch])
            best_val_acc.append(val_accuracies[best_epoch])
            best_test_acc.append(test_accuracies[best_epoch])
            best_train_acc.append(train_accuracies[best_epoch])
            best_epochs.append(best_epoch)
        best_loss = tensor(best_loss)
        best_val_acc = tensor(best_val_acc)
        best_test_acc = tensor(best_test_acc)
        best_train_acc = tensor(best_train_acc)
        best_epochs = tensor(best_epochs, dtype=torch.float)
        times_tensor = tensor(times)
        group_results[caption] = {
            "best_loss_mean": mean(best_loss).item(),
            "best_loss_std": std(best_loss).item(),
            "best_val_acc_mean": mean(best_val_acc).item(),
            "best_val_acc_std": std(best_val_acc).item(),
            "best_test_acc_mean": mean(best_test_acc).item(),
            "best_test_acc_std": std(best_test_acc).item(),
            "best_train_acc_mean": mean(best_train_acc).item(),
            "best_train_acc_std": std(best_train_acc).item(),
            "train_time_mean": mean(times_tensor).item(),
            "train_time_std": std(times_tensor).item(),
            "best_epoch_mean": mean(best_epochs).item(),
            "best_epoch_std": std(best_epochs).item(),
            "flops_per_epoch": flops_per_epoch
        }

        # rows are training runs, columns are epochs.
        losses_van = tensor(losses_van)
        val_accuracies_van = tensor(val_accuracies_van)
        test_accuracies_van = tensor(test_accuracies_van)
        train_accuracies_van = tensor(train_accuracies_van)

        losses_van_mean = mean(losses_van, dim=0)
        losses_van_sd = std(losses_van, dim=0)

        linestyle = linestyles[index]
        ax1.plot(epochs, losses_van_mean, linestyle, color='#0065a7', label=f"Loss ({caption})")
        ax1.fill_between(epochs, (losses_van_mean - losses_van_sd), (losses_van_mean + losses_van_sd), color='#0065a7', alpha=0.1)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("Loss")

        val_accuracies_van_mean = mean(val_accuracies_van, dim=0)
        test_accuracies_van_mean = mean(test_accuracies_van, dim=0)
        train_accuracies_van_mean = mean(train_accuracies_van, dim=0)
        val_accuracies_van_sd = std(val_accuracies_van, dim=0)
        test_accuracies_van_sd = std(test_accuracies_van, dim=0)
        train_accuracies_van_sd = std(train_accuracies_van, dim=0)
        ax2.plot(epochs, val_accuracies_van_mean, linestyle, label=f"Validation ({caption})", color='#66a3ca')
        ax2.fill_between(epochs, (val_accuracies_van_mean - val_accuracies_van_sd),
                         (val_accuracies_van_mean + val_accuracies_van_sd), color='#66a3ca', alpha=0.1)

        ax2.plot(epochs, test_accuracies_van_mean, linestyle, label=f"Test ({caption})", color='#dda01d')
        ax2.fill_between(epochs, (test_accuracies_van_mean - test_accuracies_van_sd),
                         (test_accuracies_van_mean + test_accuracies_van_sd), color='#dda01d', alpha=0.1)
        ax2.plot(epochs, train_accuracies_van_mean, linestyle, label=f"Training ({caption})", color='#a2acbd')
        ax2.fill_between(epochs, (train_accuracies_van_mean - train_accuracies_van_sd),
                         (train_accuracies_van_mean + train_accuracies_van_sd), color='#a2acbd', alpha=0.1)

    ax2.set_ylabel("%")
    ax2.set_ylim(0, 105)
    ax1.grid()
    if loss_range is not None:
        ax1.set_ylim(loss_range[0], loss_range[1])
    fig.legend(fontsize=fontsize, loc=legend_loc)

    return ax1, ax2, fig, group_results
