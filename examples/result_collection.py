from os import listdir
import os.path as path
from os.path import splitext
import json
from pandas import DataFrame
from typing import Dict

results_dir = "results"
result_files = listdir(results_dir)
all_results = {
    "lr": [],
    "kfac_lr": [],
    "weight_decay": [],
    "dataset": [],
    "hidden_channels": [],
    "layers": [],
    "runs": [],
    "heads": [],
    "dropout": [],
    "kfac_damping": [],
    "hessianfree_damping": [],
    "optimizer": [],
    "best_loss_mean": [],
    "best_loss_std": [],
    "best_val_acc_mean": [],
    "best_val_acc_std": [],
    "best_test_acc_mean": [],
    "best_test_acc_std": [],
    "best_train_acc_mean": [],
    "best_train_acc_std": [],
    "type": [],
}

def append_to_dict(dest: Dict, source: Dict):
    for key, value in source.items():
        dest[key].append(value)

def add_to_dict(dest: Dict, source: Dict):
    for key, value in source.items():
        dest[key] = value

for result_file in result_files:
    filename, ext = splitext(result_file)

    if ext not in [".txt", ".csv"]:
        continue

    try:
        with open(path.join(results_dir, result_file)) as file:
            results = json.loads(file.read())
        exp_type = result_file[:3]
        for optimizer, result in results["results"].items():
            result_dict = {}
            add_to_dict(result_dict, result)
            add_to_dict(result_dict, results["experiment"])
            add_to_dict(result_dict, {"optimizer": optimizer, "type": exp_type})
            append_to_dict(all_results, result_dict)
    except Exception as error:
        print(f"Error loading {filename}: {error}")
df = DataFrame(all_results)
print(df.to_markdown())

with open(path.join(results_dir, "all_results.html"), "w+") as result_file:
    result_file.write(df.to_html())

with open(path.join(results_dir, "all_results.md"), "w+") as result_file:
    result_file.write(df.to_markdown())