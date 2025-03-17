import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib


class SyntheticIrradiance:
    
    def __init__(
            self,
            times = pd.date_range(start="2025-03-01", end="2025-03-07", freq="min"),
            location = {"latitude": -41.13941227780086,
                        "longitude": -73.02542294598776,
                        "timezone": -4,
                        "name": "Frutillar"},
            irradiance_type = "ghi"
            ):
        self.times = times
        self.location = location
        self.irradiance_type = irradiance_type

        self.series = np.zeros(len(times))
        self.add_clearsky()
        """ The purpose of self.series is to provide
            easy access to the finished build of the series.

            For extended use, the class provides the components:
            - self.clearsky - Through the add_clearsky method
            - self.noise    - Through the add_noise method

            Adding outliers and sensor malfunction replace values
            in the series "irreversibly"
            - self.add_outliers
            - self.add_malfunction
        """

        self.outlier_mask = np.zeros_like(self.series, dtype=bool)
        self.malfunction_mask = np.zeros_like(self.series, dtype=bool)

    def add_clearsky(self):

        # Check if location's name was provided
        if "name" not in self.location:
            self.location["name"] = "Unknown"

        location = pvlib.location.Location(
                                self.location["latitude"],
                                self.location["longitude"],
                                self.location["timezone"],
                                name=self.location["name"])
        
        # pvlib's get_clearsky returns a dataframe of ghi, dni, dhi
        components = location.get_clearsky(self.times,
                                           model="ineichen")

        # Define the clearsky component and update
        self.clearsky = components[self.irradiance_type]
        self.series = self.series + self.clearsky
        return self.series
                                         
    def add_noise(self,
                  noise_level=0.001):
        std = noise_level * np.max(self.clearsky)
        rng = np.random.default_rng()
        
        # Define the noise component and update
        self.noise = rng.normal(loc=0,
                                scale=std,
                                size=len(self.times))
        self.series = self.series + self.noise
        return self.series
    
    def add_outliers(self,
                     outlier_rate=0.01,
                     amplifier=3):
        # Introduces outliers at random positions
        num_outliers = round(outlier_rate * len(self.series))
        rng = np.random.default_rng()
        indices = rng.choice(a=len(self.series),
                             size=num_outliers)
        
        # Prevent outliers from affecting malfunction behavior
        self.outlier_mask[indices] = True
        self.outlier_mask = self.outlier_mask & ~ self.malfunction_mask
        
        # Update series
        self.series.loc[self.outlier_mask] = np.abs(self.clearsky.loc[self.outlier_mask]
                                                * amplifier)
        return self.series

    def add_sensor_disconnect(self,
                              disconnect_ratio=0.15,
                              num_events=4,
                             ):
        pass
        # Introduces continuous linear drift to simulate sensor disconnections

        fail_length = round(len(self.series) * (disconnect_ratio / num_events))

        rng = np.random.default_rng()
        # Check indices are distanced from each other (15 attempts)
        for count in range(1,16):
            indices = rng.choice(a = len(self.series) - fail_length,
                                 size=num_events)
            indices = np.sort(indices)
            d_idx = np.diff(indices)

            if all(d_idx >= 2 * fail_length):
                break
            
        # Linear drift (starts from clearsky model)
        for index in indices:
            x0 = index
            y0 = self.clearsky.iloc[index]
            m = 0.005 * rng.random(1) * np.mean(self.clearsky)
            x = np.arange(index, index + fail_length)
            x = x[x < len(self.clearsky)]  # Prevents attempt to access inexistent values
            y = m*(x - x0) + y0

            # Update masks
            self.malfunction_mask[x] = True
            self.outlier_mask[x] = False

            # Update series
            self.series.iloc[x] = y
        return self.series