import numpy as np
import pandas as pd
import pvlib

if __name__ == "__main__":
    from utils import validate_pvlib_location, validate_timezone_aware
else:
    from . utils import validate_pvlib_location, validate_timezone_aware


def get_night_mask(times: pd.DatetimeIndex,
                   location: pvlib.location.Location) -> np.ndarray:
    """ Takes time and location and returns a mask for night timestamps.
        timeseries: pandas.Series (must be timezone aware)
        Location: pvlib.location.Location
    """
    # Check input validity:
    validate_pvlib_location(location)
    validate_timezone_aware(times)

    df = pvlib.solarposition.sun_rise_set_transit_spa(times,
                                                      location.latitude,
                                                      location.longitude,
                                                      how="numpy")
    return (times < df["sunrise"]) | (times > df["sunset"])


def anomaly_ceiling(timeseries: pd.Series, max_value: float):
    # Simply returns a mask
    return timeseries > max_value

def anomaly_clearsky(timeseries: pd.Series,
                     location: pvlib.location.Location,
                     irradiance_type: str,
                     day_margin: float = 1.25,
                     night_threshold: float = 10):
    """
    Identifies outliers in a timeseries of irradiance data by comparing it to 
    a clearsky model.

    Args:
        timeseries: The irradiance timeseries to check.
        location: The pvlib Location object representing the site.
        irradiance_type: The type of irradiance (e.g., 'ghi', 'dni', 'dhi').
        day_margin: The margin by which the timeseries can deviate from the 
            clearsky model before being considered an anomaly.
        night_threshold: A threshold value for nighttime irradiance.

    Returns:
        mask: A pandas Series with boolean values indicating anomalies.
        general_threshold: A pandas Series to plot the boundary of the algorithm
    """
    is_night = get_night_mask(timeseries.index, location)

    # pvlib's get_clearsky returns a dataframe of ghi, dni, dhi
    components = location.get_clearsky(timeseries.index, model="ineichen")
    clearsky = components[irradiance_type]
    
    # Mask of values that exceed an irradiance threshold:
    # Adjusted to compensate for lower values when irradiance is closer to 0.
    day_threshold = np.maximum(clearsky * day_margin, clearsky + 50)
    general_threshold = day_threshold
    general_threshold[is_night] = night_threshold
    
    is_day_outlier = ~is_night & (timeseries > day_threshold)
    is_night_outlier = is_night & (timeseries > night_threshold)

    mask = is_day_outlier | is_night_outlier
    return mask, general_threshold

def anomaly_linear(timeseries: pd.Series,
                   location: pvlib.location.Location,
                   horizon: int = 120,
                   tolerance: float = 1,
                   min_irradiance: float = 10):
    
    is_daytime = ~ get_night_mask(timeseries.index, location)
    is_relevant = (timeseries.values >= min_irradiance) & is_daytime

    """
    Adjust a linear curve to a rolling horizon to evaluate how good of a fit it
    is at each step. Ignores night values.
    """

    def linfit_score(series_sample) -> float:
        t = np.arange(len(series_sample))
        slope, intercept = np.polyfit(t, series_sample, 1)

        # Calculate score and return:
        x_fit = slope * t + intercept
        return np.std(series_sample - x_fit)

    # Use a moving window:
    is_linear = timeseries.rolling(horizon).apply(
        lambda s: linfit_score(s)) <= tolerance
    
    # Index corresponds to right edge, so we fill the rest of the window:
    for k in range(horizon, len(is_linear)):
        if is_linear.iloc[k]:
            is_linear.iloc[k - horizon] = True

    return is_linear & is_relevant



# Example code for debug
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import synthetic_data_generation as sdg

    ghi = sdg.SyntheticIrradiance()
    ghi.add_sensor_disconnect()
    linear_mask = anomaly_linear(ghi.series, 120, 1)


    # Test night_mask
    night_mask = get_night_mask(ghi.times, ghi.location)
    print(f"type of night_mask: {type(night_mask)}")
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
