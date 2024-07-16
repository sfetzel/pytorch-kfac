import pandas
import torch
import json
import os
base_dir = "results"
filenames = []

results = []
time_digits = 3
epochs = 200
for filename in filenames:
    with open(os.path.join(base_dir, filename), "r") as file:
        result_json = json.load(file)
        model = filename[:3]
        #epochs = result_json["experiment"]["epochs"]
        dataset = result_json["experiment"]["dataset"]
        train_adam_mean = result_json["results"]["adam"]["train_time_mean"]
        train_adam_std = result_json["results"]["adam"]["train_time_std"]
        train_kfac_mean = result_json["results"]["KFAC"]["train_time_mean"]
        train_kfac_std = result_json["results"]["KFAC"]["train_time_std"]

        results.append([
            model,
            dataset,
            round(train_adam_mean, time_digits),
            round(train_adam_std, time_digits),
            round(train_adam_mean / epochs, time_digits),
            round(train_adam_std / epochs, time_digits),
            round(train_kfac_mean, time_digits),
            round(train_kfac_std, time_digits),
            round(train_kfac_mean / epochs, time_digits),
            round(train_kfac_std / epochs, time_digits),
            round(train_kfac_mean / train_adam_mean, time_digits)
        ])
        
df = pandas.DataFrame(results, columns=[
    "model",
    "dataset",
    "adam-time-mean",
    "adam-time-std",
    "adam-time-epoch-mean",
    "adam-time-epoch-std",
    "kfac-time-mean",
    "kfac-time-std",
    "kfac-time-epoch-mean",
    "kfac-time-epoch-std",
    "kfac-ratio"
])
df.to_csv(os.path.join(base_dir, "efficiency.csv"))
df
