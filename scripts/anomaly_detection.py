import numpy as np
import pandas as pd
import pvlib

def get_night_mask(timeseries, location):
    # Takes a timeseries and pvlib.Location and returns a mask for the nighttime values
    times = timeseries.index

    # Check if the times input is timezone-aware:
    if times.tz is None:
        raise ValueError("DatetimeIndex must be timezone-aware.")

    df = pvlib.solarposition.sun_rise_set_transit_spa(times,
                                                      location.latitude,
                                                      location.longitude,
                                                      how="numpy")
    return (times < df["sunrise"]) | (times > df["sunset"])


def anomaly_ceiling(timeseries, max_value):
    # Simply returns a mask
    return timeseries > max_value

def anomaly_clearsky(timeseries,
                     location,
                     irradiance_type,
                     margin=1.2,
                     max_night_irradiance=10):
    """_summary_

    Args:
        timeseries (_pandas Series or dataframe_): Irradiance timeseries
        location (_dictionary_): Must include latitude, longitude, and timezone
        irradiance_type (_string_): "ghi", "dni" or "dhi"
        margin (float, optional): for outlier detection, defaults to 1.2.
        max_night_irradiance (int, optional): Threshold to ignor irradiance,
        Defaults to 10.

    Returns:
        mask 1 _logical_: Mask of values that are considered outliers
        irradiance_threshold _pandas.Series_: to plot the algorithm's margin
    """
    
    # Check if location's name was provided
    if "name" not in location:
        location["name"] = "Unknown"

    # Get clearsky irradiance for the location:
    pvlocation = pvlib.location.Location(
                            location["latitude"],
                            location["longitude"],
                            location["timezone"],
                            name=location["name"])
    components = pvlocation.get_clearsky(timeseries.index, model="ineichen")

    # pvlib's get_clearsky returns a dataframe of ghi, dni, dhi
    clearsky = components[irradiance_type]
    
    # Mask of values that exceed an irradiance threshold:
    # Adjusted to compensate for lower values when irradiance is closer to 0.
    irradiance_threshold = np.maximum(clearsky * margin, clearsky + 50)
    is_daytime = timeseries > max_night_irradiance
    is_outlier = timeseries > irradiance_threshold
    mask = is_daytime & is_outlier

    return mask, irradiance_threshold

def anomaly_linear(ts,
                   horizon=120,
                   tolerance=1,
                   max_night_irradiance=10,
                   linfit_accuracy=1):
    """ Adjust a linear curve to a rolling horizon
        to evaluate how good of a fit it is at each step
    """
    def linfit_score(ts):
        t = np.arange(len(ts))
        slope, intercept = np.polyfit(t, ts, 1)

        # Calculate score and return:
        x_fit = slope * t + intercept
        return np.std(ts - x_fit)

    # Note: Index corresponds to right edge of window.
    is_linear = ts.rolling(horizon).apply(lambda s: linfit_score(s)) <= tolerance
    for k in range(horizon, len(is_linear)):
        if is_linear.iloc[k]:
            is_linear.iloc[k - horizon] = True

    # Return mask:
    is_daytime = ts > max_night_irradiance
    return is_linear & is_daytime

# Example code for debug
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import synthetic_data_generation as sdg

    ghi = sdg.SyntheticIrradiance()
    ghi.add_sensor_disconnect()
    linear_mask = anomaly_linear(ghi.series, 120, 1)


    # Test night_mask
    night_mask = get_night_mask(ghi.series, ghi.location)

    # plot tests:
    plt.plot(ghi.times,
             ghi.series,
             'b.',
             markersize=2,
             label="timeseries")
    plt.plot(ghi.times[linear_mask],
             ghi.series.loc[linear_mask],
             'r.',
             markersize=2,
             label="linear anomaly")
    plt.plot(ghi.times,
             night_mask * np.max(ghi.clearsky),
             'k-',
             markersize=2,
             label="night_mask")
    plt.show()
