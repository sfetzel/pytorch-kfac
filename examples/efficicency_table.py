import pandas
import torch
import json
import os

base_dir = "results"
result_files = os.listdir(base_dir)
filenames = [filename for filename in result_files if filename.endswith(".txt") and
             ("kipf" in filename or "Veličković" in filename)]

max_flops = 10.6e12  # 10.6TFLOPs
results = []
time_digits = 3
epochs = 200
for filename in filenames:
    with open(os.path.join(base_dir, filename), "r") as file:
        result_json = json.load(file)
        model = filename[:3]
        #epochs = result_json["experiment"]["epochs"]
        dataset = result_json["experiment"]["dataset"]
        epochs = result_json["experiment"]["epochs"]
        train_adam_mean = result_json["results"]["adam"]["train_time_mean"]
        train_adam_std = result_json["results"]["adam"]["train_time_std"]
        # FLOPs / (FLOPs/s) = s.
        adam_efficiency = ""
        adam_efficiency_best_epoch = ""

        adam_min_time_per_epoch = (result_json["results"]["adam"]["flops_per_epoch"]) / max_flops

        if result_json["results"]["adam"]["flops_per_epoch"] is not None:
            best_epoch = result_json["results"]["adam"]["best_epoch_mean"]
            adam_min_time_best_epoch = adam_min_time_per_epoch * best_epoch
            adam_efficiency_best_epoch = adam_min_time_best_epoch / (train_adam_mean / epochs * best_epoch) * 100
            adam_efficiency = (adam_min_time_per_epoch * epochs) / train_adam_mean * 100

        train_kfac_mean = result_json["results"]["KFAC"]["train_time_mean"]
        train_kfac_std = result_json["results"]["KFAC"]["train_time_std"]

        epochs_for_efficiency = result_json["results"]["KFAC"]["best_epoch_mean"]
        kfac_efficiency = ""
        kfac_efficiency_best_epoch = ""

        if result_json["results"]["KFAC"]["flops_per_epoch"] is not None:
            kfac_min_time_per_epoch = (result_json["results"]["KFAC"]["flops_per_epoch"]) / max_flops
            best_epoch = result_json["results"]["KFAC"]["best_epoch_mean"]
            adam_min_time_best_epoch = kfac_min_time_per_epoch * best_epoch
            kfac_efficiency_best_epoch = adam_min_time_best_epoch / (train_kfac_mean / epochs * best_epoch) * 100
            kfac_efficiency = (kfac_min_time_per_epoch * epochs) / train_kfac_mean * 100

        results.append([
            model,
            dataset,
            round(train_adam_mean, time_digits),
            round(train_adam_std, time_digits),
            round(train_adam_mean / epochs, time_digits),
            round(train_adam_std / epochs, time_digits),
            round(adam_efficiency, time_digits),
            round(adam_efficiency_best_epoch, time_digits),
            round(train_kfac_mean, time_digits),
            round(train_kfac_std, time_digits),
            round(train_kfac_mean / epochs, time_digits),
            round(train_kfac_std / epochs, time_digits),
            round(train_kfac_mean / train_adam_mean, time_digits),
            round(kfac_efficiency, time_digits) if kfac_efficiency != "" else "",
            round(kfac_efficiency_best_epoch, time_digits) if kfac_efficiency_best_epoch != "" else "",
        ])

df = pandas.DataFrame(results, columns=[
    "model",
    "dataset",
    "adam-time-mean",
    "adam-time-std",
    "adam-time-epoch-mean",
    "adam-time-epoch-std",
    "adam-efficiency",
    "adam-efficiency-best-epoch",
    "kfac-time-mean",
    "kfac-time-std",
    "kfac-time-epoch-mean",
    "kfac-time-epoch-std",
    "kfac-ratio",
    "kfac-efficiency",
    "kfac-efficiency-best-epoch",
])
df.to_csv(os.path.join(base_dir, "efficiency.csv"))
df
