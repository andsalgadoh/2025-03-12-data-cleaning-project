import numpy as np
import pandas as pd
import pvlib

# El código debe identificar outliers, valores extremadamente más altos de lo esperable.
# Hacer ejemplos sin irradiancia clear-sky
# Hacer ejemplo con irradiancia clear-sky (requiere conocer ubicación geográfica)
# Hacer detector de Linear drift

def anomaly_ceiling(timeseries, max_value):
    # Simply returns a mask
    return timeseries > max_value

def anomaly_clearsky(timeseries,
                     location,
                     irradiance_type,
                     margin=1.25):
    
    # Check if location's name was provided
    if "name" not in location:
        location["name"] = "Unknown"

    location = pvlib.location.Location(
                            location["latitude"],
                            location["longitude"],
                            location["timezone"],
                            name=location["name"])
    
    # pvlib's get_clearsky returns a dataframe of ghi, dni, dhi
    components = location.get_clearsky(timeseries.index, model="ineichen")

    # Define the clearsky component and update
    clearsky = components[irradiance_type]

    # Mask of values that exceed clear-sky irradiance by a margin
    return  timeseries > clearsky * margin

def anomaly_disconnection(timeseries,
                          horizon=500,
                          tolerance=1e-2,
                          max_night_irradiance=0.05):
    
    X = timeseries
    dX = timeseries.diff()  # dX(k) = X(k) - X(k-1)
    ddX = dX.diff()  # ddX(k) = dX(k) - dX(k-1) = X(k) - X(k-1) - X(k-1) + X(k-2) = X(k) -2X(k-1) + X(k-2)

    # Note: Index corresponds to right edge of window.
    is_linear = ddX.rolling(horizon).max() <= tolerance
    for k in range(horizon, len(is_linear)):
        if is_linear.iloc[k]:
            is_linear.iloc[k - horizon] = True

    is_day_intensity = X > max_night_irradiance

    # Returm mask:
    return is_linear & is_day_intensity

# Example code for debug
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import synthetic_data_generation as sdg

    ghi = sdg.SyntheticIrradiance()
    ghi.add_sensor_disconnect()
    mask = anomaly_disconnection(ghi.series, 500, 1e-2)

    plt.plot(ghi.times, ghi.series, '.')
    plt.plot(ghi.times[mask], ghi.series.loc[mask], 'o', markersize=2)
    plt.show()