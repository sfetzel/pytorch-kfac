import json

from pandas import DataFrame, read_csv
from os import listdir
from os.path import join

results_dir = "results"
best_results = []
result_files = listdir(results_dir)

for result_filename in result_files:
    if result_filename.endswith("-ggn.txt") or result_filename.endswith("-hessian.txt"):
        filepath = join(results_dir, result_filename)
        try:
            with open(filepath, "r") as result_file:
                df = read_csv(filepath, delimiter=';')
                parts = result_filename.split('-')
                model = parts[0]
                dataset = parts[1]

                results = json.load(result_file)
                kfac_results = results["results"]["K-FAC"]
                
                if "GGN" in results["results"]:
                    other_results = results["results"]["GGN"]
                    type = "GGN"
                else:
                    other_results = results["results"]["Hessian"]
                    type = "Hessian"
                config = results["experiment"]
                best_results.append([
                    model,
                    dataset,
                    type,
                    config["dropout"],
                    config["kfac_damping"],
                    config["lr"],
                    config["weight_decay"],
                    config["cov_update_freq"],
                    config["cov_ema_decay"],
                    round(other_results["best_train_acc_mean"], 3),
                    round(other_results["best_train_acc_std"], 3),
                    round(other_results["best_val_acc_mean"], 3),
                    round(other_results["best_val_acc_std"], 3),
                    round(other_results["best_test_acc_mean"], 3),
                    round(other_results["best_test_acc_std"], 3),
                    round(other_results["best_loss_mean"], 3),
                    round(other_results["best_loss_std"], 3),
                    round(kfac_results["best_test_acc_mean"], 3),
                    round(kfac_results["best_test_acc_std"], 3),
                    round(kfac_results["best_loss_mean"], 3),
                    round(kfac_results["best_loss_std"], 3),
                ])
                print(other_results)
        except Exception as error:
            print(f"Could not read {result_filename}: {error}")

df = (DataFrame(best_results, columns=["model", "dataset", "type", "dropout", "kfac_damping", "lr",
                                      "weight_decay", "cov_update_freq", "cov_ema_decay",
                                      "train_acc_mean", "train_acc_std", "val_acc_mean", "val_acc_std",
                                      "test_acc_mean", "test_acc_std",
                                      "loss_mean", "loss_std",
                                      "kfac_test_acc_mean", "kfac_test_acc_std",
                                      "kfac_loss_mean", "kfac_loss_std",], index=None)
      .sort_values(by=["dataset", "model", "type"]))

print((df["kfac_test_acc_mean"] - df["test_acc_mean"]).abs().describe())
df.to_csv("results/hessianfree.csv", index=False)
