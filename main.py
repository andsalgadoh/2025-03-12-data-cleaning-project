import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scripts.synthetic_data_generation as sdg
# from scripts.load_data import load_dataset
# from utils import detect_anomalies

# Generate synthetic irradiance data:
times = pd.date_range(start="2025-03-01", end="2025-03-07", freq="min")
clearsky_irradiance = sdg.GenerateIrradiance.clearsky(times)
ghi = clearsky_irradiance["ghi"]

# Visualize synthetic irradiance data:
plt.figure(1)
plt.plot(times, ghi)
plt.show()

# Add gausian noise to the data and visualize it
noisy_ghi = sdg.GenerateIrradiance.add_noise(ghi)
plt.figure(1)
plt.plot(times, noisy_ghi)
plt.show()

# Add extreme outliers to the data
# df = load_dataset("data/private/private_data_raw.csv")

# # t= df["Time"].iloc[1440:1440*7]
# # power = df["power"].iloc[1440:1440*7]
# # temperature = df["temperature"].iloc[1440:1440*7]

# t= df["Time"]
# power = df["power"]
# temperature = df["temperature"]

# plt.figure(figsize=(10,5))
# plt.plot(t, power)
# plt.plot(t, temperature)
# plt.xlabel("Time")
# plt.ylabel("Power")
# plt.title("Power over Time")
# plt.xticks([])
# plt.show()

# # plt.plot(df["date"], df["power"])
# # plt.xlabel("Time")
# # plt.ylabel("Power")
# # plt.title("Power over Time")
# # plt.show()


# # # Example dataset
# # data = [25, 26, 1000, 27, 28, -500, 29]
# # anomalies = detect_anomalies(data, tol=1.0)

# # print(anomalies)  # Boolean mask for anomalies