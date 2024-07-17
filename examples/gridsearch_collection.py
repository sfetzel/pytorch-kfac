from pandas import DataFrame, read_csv
from os import listdir
from os.path import join

results_dir = "results"
best_results = []
for result_filename in listdir(results_dir):
    if result_filename.endswith("gridsearch.csv"):
        filepath = join(results_dir, result_filename)
        try:
            with open(filepath, "r") as result_file:
                df = read_csv(filepath, delimiter=';')
                parts = result_filename.split('_')
                model = parts[0]
                dataset = parts[1]
                print(f"File: {result_filename}, Best test acc config:")
                best_test_config = df.iloc[df["test_acc_mean"].idxmax()]
                #print(best_test_config)

                print(f"File: {result_filename}, Best val acc config:")
                best_val_config = df.iloc[df["val_acc_mean"].idxmax()]
                best_results.append([
                    model,
                    dataset,
                    best_val_config["dropout"],
                    best_val_config["kfac_damping"],
                    best_val_config["lr"],
                    best_val_config["weight_decay"],
                    best_val_config["cov_update_freq"],
                    best_val_config["cov_ema_decay"],
                    round(best_val_config["train_acc_mean"], 3),
                    round(best_val_config["train_acc_std"], 3),
                    round(best_val_config["val_acc_mean"], 3),
                    round(best_val_config["val_acc_std"], 3),
                    round(best_val_config["test_acc_mean"], 3),
                    round(best_val_config["test_acc_std"], 3),
                    round(best_val_config["loss_mean"], 3),
                    round(best_val_config["loss_std"], 3),

                ])
                print(best_val_config)
        except Exception as error:
            print(f"Could not read {result_filename}: {error}")

df = DataFrame(best_results, columns=["model", "dataset", "dropout", "kfac_damping", "lr",
                                      "weight_decay", "cov_update_freq", "cov_ema_decay",
                                      "train_acc_mean", "train_acc_std", "val_acc_mean", "val_acc_std",
                                      "test_acc_mean", "test_acc_std",
                                      "loss_mean", "loss_std"], index=None).sort_values(by=["model", "dataset"])
df.to_csv("results/gridsearch.csv", index=False)
