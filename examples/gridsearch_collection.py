from pandas import DataFrame, read_csv
from os import listdir
from os.path import join

results_dir = "results"

for result_filename in listdir(results_dir):
    if result_filename.endswith("gridsearch.csv"):
        filepath = join(results_dir, result_filename)
        try:
            with open(filepath, "r") as result_file:
                df = read_csv(filepath, delimiter=';')
                print(f"File: {result_filename}, Best test acc config:")
                best_test_config = df.iloc[df["test_acc_mean"].idxmax()]
                #print(best_test_config)

                print(f"File: {result_filename}, Best val acc config:")
                best_val_config = df.iloc[df["val_acc_mean"].idxmax()]
                print(best_val_config)
        except Exception as error:
            print(f"Could not read {result_filename}: {error}")