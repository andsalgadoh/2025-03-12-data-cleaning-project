import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib

class GenerateIrradiance:
    
    def clearsky(times,
            latitude = -41.13941227780086,
            longitude = -73.02542294598776,
            tz = -4,
            name = "Frutillar"
    ):
        location = pvlib.location.Location(latitude,
                                           longitude,
                                           tz,
                                           name=name)
        # Function returns a dataframe of ghi, dni, dhi
        return location.get_clearsky(times,
                                     model="ineichen")
                                         
    def add_noise(irradiance_ts, noise_level=0.02):
        # Adds random noise to an irradiance timeseries
        std = noise_level * np.mean(irradiance_ts)
        rng = np.random.default_rng()
        noise = rng.normal(loc=0,
                           scale=std,
                           size=len(irradiance_ts))
        return irradiance_ts + noise
    
    def add_extreme_outliers(
            irradiance_ts,
            outlier_rate=0.01,
            value_multiplier=5):
        # Introduces extreme outliers at random positions
        rng = np.random.default_rng()
        length = len(irradiance_ts)
        num_outliers = round(length * outlier_rate)
        indices = rng.choice(a=length,
                             size=num_outliers
                             )
        irradiance_ts.iloc[indices] = np.abs(irradiance_ts.iloc[indices] * value_multiplier)
        return irradiance_ts

    def add_sensor_disconnect(
            irradiance_ts,
            total_fail_rate=0.1,
            num_events=5,
            ):
        # Introduces continuous linear drift to simulate sensor disconnections

        length = len(irradiance_ts)
        fail_length = round(length * (total_fail_rate / num_events))

        rng = np.random.default_rng()
        indices = rng.choice(a=length,
                             size=num_events
                             )  # indices is an ndarray
        
        # Linear drift
        for index in indices:
            x0 = index
            y0 = irradiance_ts.iloc[index]
            m = 0.01 * rng.random(1) * np.mean(irradiance_ts)

            x = np.arange(index, index + fail_length)
            y = m*(x - x0) + y0
            irradiance_ts.iloc[x] = y
        
        return irradiance_ts