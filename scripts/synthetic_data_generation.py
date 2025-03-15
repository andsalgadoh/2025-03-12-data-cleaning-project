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
                                         
    def add_noise(irradiance_timeseries):
        print("Function hasn't been coded yet")
    
    def add_sensor_disconnect(irradiance_timeseries):
        print("Function hasn't been coded yet")

    def add_extreme_outliers(irradiance_timeseries):
        print("Function hasn't been coded yet")
