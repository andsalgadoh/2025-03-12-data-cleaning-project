import numpy as np
import pandas as pd
import pvlib

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

def anomaly_linear(timeseries,
                   horizon=120,
                   tolerance=1,
                   max_night_irradiance=10):
    """ Adjust a linear curve to a rolling horizon
        to evaluate how good of a fit it is at each step
    """
    def linfit_score(ts):
        t = np.arange(len(ts))
        slope, intercept = np.polyfit(t, ts, 1)
        x_fit = slope * t + intercept
        return np.std(ts - x_fit)

    X = timeseries
    X.rolling(horizon).apply(lambda s: linfit_score(s))

    # Note: Index corresponds to right edge of window.
    is_linear = X.rolling(horizon).apply(lambda s: linfit_score(s)) <= tolerance
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
    mask = anomaly_linear(ghi.series, 120, 1)

    plt.plot(ghi.times, ghi.series, '.')
    plt.plot(ghi.times[mask], ghi.series.loc[mask], '.', markersize=2)
    plt.show()