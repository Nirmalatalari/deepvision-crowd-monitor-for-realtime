# ex.py
import numpy as np
import pandas as pd
imgs = np.load("images.npy", allow_pickle=False)
labels = np.load("labels.npy", allow_pickle=False)
print("Images.npy shape:", imgs.shape, imgs.dtype)
print("Labels.npy shape:", labels.shape, labels.dtype)

import pandas as pd
try:
    csv = pd.read_csv("labels.csv")
    print(csv.head())
except Exception:
    pass
