import os
import tempfile
import pandas as pd 

for root, _, files in os.walk("CSV"):
    for file in files:
        if file.endswith(".csv"):
            data = pd.read_csv(os.path.join(root, file))

            files = data.iloc[:, 0].tolist()
            for f in files:
                if not os.path.isfile(f):
                    print(f"Missing file: {f} in CSV: {os.path.join(root, file)}")