from matplotlib import pyplot
from torch import tensor, mean, std


def plot_training(epoch_count, runs, training_fun, parameter_groups: list, captions: list):
    times = []
    best_val_acc = final_test_acc = 0
    best_loss = 0
    epochs = range(1, epoch_count + 1)

    fig, ax1 = pyplot.subplots()
    ax2 = ax1.twinx()
    linestyles = ["-", "--"]

    for index, (parameter_group, caption) in enumerate(zip(parameter_groups, captions)):
        losses_van, val_accuracies_van, test_accuracies_van, train_accuracies_van = [], [], [], []
        for i in range(runs):
            losses, val_accuracies, test_accuracies, train_accuracies = training_fun(parameter_group)
            losses_van.append(losses)
            val_accuracies_van.append(val_accuracies)
            test_accuracies_van.append(test_accuracies)
            train_accuracies_van.append(train_accuracies)

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

    ax2.grid()
    fig.legend(loc='upper left')
    return ax1, ax2, fig
