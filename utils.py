import pandas as pd
import os


def load_data(root, names):
    print(f"Loading data from {root}, with data: {names}")
    engine = "pyarrow"
    if isinstance(names, str):
        return pd.read_csv(os.path.join(root, f"{names}.csv"), engine=engine)
    data = []
    for name in names:
        data.append(pd.read_csv(os.path.join(root, f"{name}.csv"), engine=engine))
    return data


def save_data(root, data):
    print(f"Saving data to {root}, with data: {list(data.keys())}")
    os.makedirs(root, exist_ok=True)
    for name, df in data.items():
        df.to_csv(os.path.join(root, f"{name}.csv"), index=False, float_format="%.6f")
