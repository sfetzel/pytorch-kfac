import json

from pandas import DataFrame, read_csv
from os import listdir
from os.path import join

results_dir = "results"
best_results = []
result_files = listdir(results_dir)
gridsearch_suffix = "gridsearch.csv"
for result_filename in result_files:
    if result_filename.endswith(gridsearch_suffix):
        filepath = join(results_dir, result_filename)
        try:
            with open(filepath, "r") as result_file:
                df = read_csv(filepath, delimiter=';')
                parts = result_filename.split('_')
                model = parts[0]
                dataset = parts[1]

                # look for existing results from experiments with
                # hyperparameters from "Kipf" and "Veličković".
                existing_results = [file for file in result_files
                                    if not file.endswith(gridsearch_suffix) and
                                    file.startswith(f"{model}-{dataset}") and
                                    file.endswith("txt") and
                                    ("kipf" in file or "Veličković" in file)]

                if existing_results:
                    print(f"Found existing results for {model}-{dataset}: {existing_results[0]}")
                    results = json.load(open(join(results_dir, existing_results[0]), "r"))
                    adam_results = results["results"]["ADAM"]
                    # original KFAC results without hyperparameter gridsearch.
                    kfac_results = results["results"]["K-FAC"]
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
                        round(adam_results["best_test_acc_mean"], 3),
                        round(adam_results["best_test_acc_std"], 3),
                        round(adam_results["best_loss_mean"], 3),
                        round(adam_results["best_loss_std"], 3),
                        round(kfac_results["best_test_acc_mean"], 3),
                        round(kfac_results["best_test_acc_std"], 3),
                        round(kfac_results["best_loss_mean"], 3),
                        round(kfac_results["best_loss_std"], 3),
                    ])
                    print(best_val_config)
        except Exception as error:
            print(f"Could not read {result_filename}: {error}")

df = (DataFrame(best_results, columns=["model", "dataset", "dropout", "kfac_damping", "lr",
                                      "weight_decay", "cov_update_freq", "cov_ema_decay",
                                      "train_acc_mean", "train_acc_std", "val_acc_mean", "val_acc_std",
                                      "test_acc_mean", "test_acc_std",
                                      "loss_mean", "loss_std",
                                      "adam_test_acc_mean", "adam_test_acc_std",
                                      "adam_loss_mean", "adam_loss_std",
                                      "kfac_test_acc_mean", "kfac_test_acc_std",
                                      "kfac_loss_mean", "kfac_loss_std",], index=None)
      .sort_values(by=["dataset", "model"]))
df.to_csv("results/gridsearch.csv", index=False)
