import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.load_data import load_dataset
from utils import detect_anomalies

df = load_dataset("data/private/dataset_pv_die_raw.csv")

t= df["Time"].iloc[1440:1440*7]
power = df["power"].iloc[1440:1440*7]
temperature = df["temperature"].iloc[1440:1440*7]

plt.figure(figsize=(10,5))
plt.plot(t, power)
plt.plot(t, temperature)
plt.xlabel("Time")
plt.ylabel("Power")
plt.title("Power over Time")
plt.xticks([])
plt.show()

# plt.plot(df["date"], df["power"])
# plt.xlabel("Time")
# plt.ylabel("Power")
# plt.title("Power over Time")
# plt.show()


# # Example dataset
# data = [25, 26, 1000, 27, 28, -500, 29]
# anomalies = detect_anomalies(data, tol=1.0)

# print(anomalies)  # Boolean mask for anomaliesc