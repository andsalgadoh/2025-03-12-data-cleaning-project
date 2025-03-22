import numpy as np
import pandas as pd
import pvlib

class SyntheticIrradiance:

    def __init__(
            self,
            times = pd.date_range(start="2025-03-01",
                                  end="2025-03-07",
                                  freq="min",
                                  tz="America/Santiago",
                                  ),
            location = pvlib.location.Location(latitude=-41.13941227780086,
                                               longitude=-73.02542294598776,
                                               tz="America/Santiago",
                                               name="Frutillar"
                                               ),
            irradiance_type = "ghi"):
        
        # Check if the times input is timezone-aware:
        if times.tz is None:
            raise ValueError("DatetimeIndex must be timezone-aware.")
        
        # Initialize values:
        self.times = times
        self.location = location
        self.irradiance_type = irradiance_type

        # Initialize series and anomaly masks:
        self.series = pd.Series(np.zeros(len(times)), index=times)
        self.add_clearsky()

        self.outlier_mask = np.zeros_like(self.series, dtype=bool)
        self.malfunction_mask = np.zeros_like(self.series, dtype=bool)

        """ The purpose of self.series is to provide
            easy access to the finished build of the timeseries.

            For extended use, the class provides the components:
            - self.clearsky - Through the add_clearsky method
            - self.noise    - Through the add_noise method

            Adding outliers and sensor malfunction replace values
            in the series and modifies their respective masks.
            - self.add_outliers
            - self.add_malfunction
        """

    def add_clearsky(self):
        # pvlib's get_clearsky returns a dataframe of ghi, dni, dhi
        components = self.location.get_clearsky(self.times, model="ineichen")

        # Define the clearsky component and update
        self.clearsky = components[self.irradiance_type]
        self.series = self.series + self.clearsky
        return
                                         
    def add_noise(self, noise_level=0.001):
        std = noise_level * np.max(self.clearsky)
        rng = np.random.default_rng()
        
        # Define the noise component and update
        self.noise = rng.normal(loc=0,
                                scale=std,
                                size=len(self.times))
        self.series = self.series + self.noise
        return
    
    def add_outliers(self, outlier_cap=3.0, outlier_percentage=0.01):
        # Introduce outliers at random positions
        rng = np.random.default_rng()
        num_outliers = round(outlier_percentage * len(self.series))
        indices = rng.choice(a=len(self.series), size=num_outliers)
        
        # Prevent outliers from affecting malfunction behavior
        self.outlier_mask[indices] = True
        self.outlier_mask = self.outlier_mask & ~ self.malfunction_mask
        
        # Set the irradiance value of the outliers:
        random_factors = rng.random(self.outlier_mask.sum())
        candidate_outliers = (np.abs(self.clearsky.loc[self.outlier_mask])
                              * (1.25 + random_factors * (outlier_cap - 1.25)))
        min_outlier_threshold = random_factors * 100 + 50

        outlier_values = np.maximum(candidate_outliers, min_outlier_threshold)
        
        # Update the timeseries:
        self.series.loc[self.outlier_mask] = outlier_values
        return

    def add_sensor_disconnect(self,
                              disconnect_ratio=0.15,
                              num_events=4):
        
        # Introduces continuous linear drift to simulate sensor disconnections
        fault_duration = round(len(self.series)
                               * (disconnect_ratio / num_events))

        # Get indices for each event ensuring they are distanced from each other
        indices = []
        attempts = 0
        max_attempts = 15  # Limit to 15 continuous failed attempts.
        rng = np.random.default_rng()

        while len(indices) < num_events and attempts < max_attempts:
            candidate = rng.integers(0, len(self.series) - fault_duration)
            if all(abs(candidate - i) >= 2 * fault_duration for i in indices):
                indices.append(candidate)
                attempts = 0
            else:
                attempts += 1
        indices.sort()
        
        # Log warning if we couldn't place all events
        if len(indices) < num_events:
            print("Warning: 'add_sensor_disconnect()'")
            print(f"only placed {len(indices)} out of {num_events} events.")

        # Linear drift (starts from clearsky model)
        # Define max slope as raising irradiance from 0 to max in 12 hours.
        step_time = self.clearsky.index[1] - self.clearsky.index[0]
        num_steps_in_12h = pd.Timedelta(12, "h") / step_time
        max_slope = np.max(self.clearsky)/(num_steps_in_12h)

        for index in indices:
            x0 = index
            y0 = self.clearsky.iloc[index]

            slope = max_slope * rng.random(1)

            # Create linear drift series:
            x = np.arange(index, index + fault_duration)
            x = x[x < len(self.clearsky)]  # Prevent access to inexistent values
            y = slope*(x - x0) + y0

            # Update masks
            self.malfunction_mask[x] = True
            self.outlier_mask[x] = False

            # Update series
            self.series.iloc[x] = y
        return self.series

if __name__ == '__main__':
    # Test script:
    import matplotlib.pyplot as plt

    ghi = SyntheticIrradiance()
    ghi.add_outliers()
    ghi.add_noise()
    ghi.add_outliers()
    ghi.add_sensor_disconnect()
    ghi.add_outliers()

    plt.plot(ghi.times,
             ghi.clearsky,
             'b-',
             linewidth=0.5,
             label="clearsky")

    plt.plot(ghi.times,
             ghi.series,
             'r.',
             markersize=1,
             label="modified")
    
    plt.legend()
    plt.show()