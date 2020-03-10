import numpy as np
import os
import shutil
import pandas as pd

files = [f for f in os.listdir('./full_history')]

# select 0.1 of the files randomly
random_files = np.random.choice(files, 100)

for i in random_files:
    shutil.copy('./full_history/' + i, "./data")

combined_csv = pd.concat(
    [
        pd.read_csv(
            "./data/" +
            f)["close"].rename(
                f.replace(
                    ".csv",
                    "")) for f in random_files],
    axis=1)

for i in combined_csv:
    if combined_csv[i][:1000].isna().any():
        del combined_csv[i]

combined_csv = combined_csv[:1000]


sdf = pd.read_csv("./data/" + random_files[0])["date"]

sdf = sdf[:1000]

combined_csv = combined_csv.set_index(sdf)

combined_csv = combined_csv.iloc[:, : 50]

combined_csv = combined_csv.iloc[::-1]

combined_csv.to_csv("amex_rand50.csv", encoding='utf-8-sig')
