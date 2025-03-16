import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scripts.synthetic_data_generation as sdg
# from scripts.load_data import load_dataset
# from utils import detect_anomalies

# Generate synthetic irradiance data:
times = pd.date_range(start="2025-03-01", end="2025-03-07", freq="min")
ghi = sdg.SyntheticIrradiance(times)  # Initialized with a clearsky model

# Visualize synthetic irradiance data:
clearsky_ghi = ghi.series

print("Visualizing synthetic clearsky irradiance")
plt.figure(1)
plt.subplot(4,1,1)
plt.title("Synthetic clearsky irradiance")
plt.plot(times, clearsky_ghi)
plt.show(block=False)

# Add sensor malfunction examples to the data and visualize:
sensor_ghi = ghi.add_sensor_disconnect()

print("Visualizing sensor malfunction")
plt.figure(1)
plt.subplot(4,1,2)
plt.title("Adding sensor malfunction")
plt.plot(times, sensor_ghi, ".", markersize=2)
plt.tight_layout()
plt.show(block=False)

# Add gausian noise to the data and visualize everything:
noisy_ghi = ghi.add_noise()

print("Visualizing noisy synthetic irradiance")
plt.figure(1)
plt.subplot(4,1,3)
plt.title("Adding noise")
plt.plot(times, noisy_ghi, ".", markersize=2)
plt.show(block=False)

# Add extreme outliers to the data and visualize:
realistic_ghi = ghi.add_outliers()

print("Visualizing extreme outliers")
plt.figure(1)
plt.subplot(4,1,4)
plt.title("Adding outliers")
plt.plot(times, realistic_ghi, ".", markersize=2)
plt.show(block=False)


# Save synthetic data to .csv file
dataset_name = "data/public/realistic_ghi_data.csv"
if not os.path.isfile(dataset_name):
    print(f"A synthetic irradiance dataset {dataset_name} does NOT exist.")
    save_confirm = input("Would you like to save this one? (Y/N): ")
else:
    print(f"A synthetic irradiance dataset already exist in {dataset_name}")
    save_confirm = input("Do you want to overwrite it? (Y/N): ").upper()

if save_confirm == "Y":
    print(f"Saving dataset in {dataset_name}")
    df = pd.DataFrame({"Timestamp": times, "GHI": realistic_ghi})
    df.set_index("Timestamp", inplace=True)
    df.to_csv(dataset_name)