import gc

import torch
from matplotlib import pyplot
from torch import tensor, mean, std, argmax

from os.path import exists, splitext

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


def plot_training(epoch_count, runs, training_fun, parameter_groups: list, captions: list):
    times = []
    best_val_acc = final_test_acc = 0
    best_loss = 0
    epochs = range(1, epoch_count + 1)
    fontsize = 25
    fig, ax1 = pyplot.subplots(figsize=(15, 10))
    ax2 = ax1.twinx()
    set_fontsize(ax1, fontsize)
    set_fontsize(ax2, fontsize)

    linestyles = ["-", "--"]
    group_results = {}
    for index, (parameter_group, caption) in enumerate(zip(parameter_groups, captions)):
        losses_van, val_accuracies_van, test_accuracies_van, train_accuracies_van = [], [], [], []

        best_loss, best_val_acc, best_test_acc, best_train_acc = [], [], [], []
        for i in range(runs):
            losses, val_accuracies, test_accuracies, train_accuracies = training_fun(parameter_group)
            gc.collect()
            torch.cuda.empty_cache()
            losses_van.append(losses)
            val_accuracies_van.append(val_accuracies)
            test_accuracies_van.append(test_accuracies)
            train_accuracies_van.append(train_accuracies)

            best_epoch = argmax(tensor(val_accuracies))
            best_loss.append(losses[best_epoch])
            best_val_acc.append(val_accuracies[best_epoch])
            best_test_acc.append(test_accuracies[best_epoch])
            best_train_acc.append(train_accuracies[best_epoch])
        best_loss = tensor(best_loss)
        best_val_acc = tensor(best_val_acc)
        best_test_acc = tensor(best_test_acc)
        best_train_acc = tensor(best_train_acc)
        group_results[caption] = {
            "best_loss_mean": mean(best_loss).item(),
            "best_loss_std": std(best_loss).item(),
            "best_val_acc_mean": mean(best_val_acc).item(),
            "best_val_acc_std": std(best_val_acc).item(),
            "best_test_acc_mean": mean(best_test_acc).item(),
            "best_test_acc_std": std(best_test_acc).item(),
            "best_train_acc_mean": mean(best_train_acc).item(),
            "best_train_acc_std": std(best_train_acc).item(),
        }

        # rows are training runs, columns are epochs.
        losses_van = tensor(losses_van)
        val_accuracies_van = tensor(val_accuracies_van)
        test_accuracies_van = tensor(test_accuracies_van)
        train_accuracies_van = tensor(train_accuracies_van)

        losses_van_mean = mean(losses_van, dim=0)
        losses_van_sd = std(losses_van, dim=0)

        linestyle = linestyles[index]
        ax1.plot(epochs, losses_van_mean, linestyle, color='blue', label=f"Loss ({caption})")
        ax1.fill_between(epochs, (losses_van_mean - losses_van_sd), (losses_van_mean + losses_van_sd), color='blue', alpha=0.1)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("Loss")

        val_accuracies_van_mean = mean(val_accuracies_van, dim=0)
        test_accuracies_van_mean = mean(test_accuracies_van, dim=0)
        train_accuracies_van_mean = mean(train_accuracies_van, dim=0)
        val_accuracies_van_sd = std(val_accuracies_van, dim=0)
        test_accuracies_van_sd = std(test_accuracies_van, dim=0)
        train_accuracies_van_sd = std(train_accuracies_van, dim=0)
        ax2.plot(epochs, val_accuracies_van_mean, linestyle, label=f"Validation ({caption})", color='orange')
        ax2.fill_between(epochs, (val_accuracies_van_mean - val_accuracies_van_sd),
                         (val_accuracies_van_mean + val_accuracies_van_sd), color='orange', alpha=0.1)

        ax2.plot(epochs, test_accuracies_van_mean, linestyle, label=f"Test ({caption})", color='green')
        ax2.fill_between(epochs, (test_accuracies_van_mean - test_accuracies_van_sd),
                         (test_accuracies_van_mean + test_accuracies_van_sd), color='green', alpha=0.1)
        ax2.plot(epochs, train_accuracies_van_mean, linestyle, label=f"Training ({caption})", color='red')
        ax2.fill_between(epochs, (train_accuracies_van_mean - train_accuracies_van_sd),
                         (train_accuracies_van_mean + train_accuracies_van_sd), color='red', alpha=0.1)

    ax2.set_ylabel("%")
    ax2.set_ylim(0, 105)
    ax2.grid()
    ax1.grid()
    ax1.set_ylim(0.0, 2.0)
    fig.legend(fontsize=fontsize, loc='lower right')

    return ax1, ax2, fig, group_results
