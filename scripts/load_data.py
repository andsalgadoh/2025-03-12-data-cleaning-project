# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

import pandas as pd

def load_dataset(filepath):
    """Loads dataset from a CSV file into a Pandas DataFrame."""
    return pd.read_csv(filepath)

if __name__ == "__main__":
    
    print("\n\n\n\n\n")

    df = load_dataset("data/private/dataset_pv_die_raw.csv")
    print(df.head(1440))  # Show first few rows
