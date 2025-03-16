import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scripts.synthetic_data_generation as sdg
# from scripts.load_data import load_dataset
# from utils import detect_anomalies

# Generate synthetic irradiance data:
times = pd.date_range(start="2025-03-01", end="2025-03-07", freq="min")
clearsky_irradiance = sdg.GenerateIrradiance.clearsky(times)
clearsky_ghi = clearsky_irradiance["ghi"]

# Visualize synthetic irradiance data:
print("Visualizing synthetic clearsky irradiance")
plt.figure(1)
plt.subplot(4,1,1)
plt.title("Synthetic clearsky irradiance")
plt.plot(times, clearsky_ghi)
plt.show(block=False)

# Add extreme outliers to the data and visualize:
print("Visualizing extreme outliers")
extreme_ghi = sdg.GenerateIrradiance.add_extreme_outliers(clearsky_ghi)
plt.figure(1)
plt.subplot(4,1,2)
plt.title("Adding outliers")
plt.plot(times, extreme_ghi, ".", markersize=2)
plt.show(block=False)

# Add sensor malfunction examples to the data and visualize:
print("Visualizing sensor malfunction")
sensor_ghi = sdg.GenerateIrradiance.add_sensor_disconnect(extreme_ghi)
plt.figure(1)
plt.subplot(4,1,3)
plt.title("Adding sensor malfunction")
plt.plot(times, sensor_ghi, ".", markersize=2)
plt.tight_layout()
plt.show(block=False)

# Add gausian noise to the data and visualize everything:
print("Visualizing noisy synthetic irradiance")
realistic_ghi = sdg.GenerateIrradiance.add_noise(sensor_ghi)
plt.figure(1)
plt.subplot(4,1,4)
plt.title("Realistic clear-sky irradiance (adding noise)")
plt.plot(times, realistic_ghi, ".", markersize=2)
plt.show(block=True)

# Save synthetic data to .csv file
dataset_name = "data/public/realistic_ghi_data.csv"
if not os.path.isfile(dataset_name):
    print(f"A synthetic irradiance dataset {dataset_name} does NOT exist.")
    save_confirm = input("Would you like to save this one? (Y/N): ")
else:
    print(f"A synthetic irradiance dataset already exist in {dataset_name}")
    save_confirm = input("Do you want to overwrite it? (Y/N): ")

if save_confirm == "Y":
    print(f"Saving dataset in {dataset_name}")
    df = pd.DataFrame({"Timestamp": times, "GHI": realistic_ghi})
    df.set_index("Timestamp", inplace=True)
    df.to_csv(dataset_name)

# Notes: "realistic_ghi" is of type pandas.core.series.Series
#        "times" is of type DatetimeIndex





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