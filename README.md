# WORK IN PROGRESS

This is part of my personal Python practice.

The code here is meant to:
1) Show how a certain algorithm can be used to identify and remove incorrect values, caused by sensor malfunction, from a dataset.
2) Evaluate the impact of data cleaning on time series forecast accuracy using simple neural network models.

My goals for this project are:
1) Get familiarized with Python and its libraries: numpy, pandas, matplotlib, pytorch.
2) Build interactive visualizations for my data-cleaning algorithms. I'm specially interested in viewing the effect of tuning a certain parameter in real time, to see what data points are chosen for each tolerance value.

# Scripts Folder:
## anomaly_detection.py
### function: get_night_mask()
Takes time and location and returns a mask for night timestamps, it makes use of pvlib's sun_rise_set_transit_spa() function:\
https://pvlib-python.readthedocs.io/en/v0.6.1/generated/pvlib.solarposition.sun_rise_set_transit_spa.html

### function: anomaly_clearsky()
Identifies outliers in a timeseries of irradiance data by comparing it to 
a clearsky model.

Args:\
  **timeseries**: The irradiance timeseries to check.\
  **location**: The pvlib Location object representing the site.\
  **irradiance_type**: The type of irradiance (e.g., 'ghi', 'dni', 'dhi').\
  **day_margin**: The margin by which the timeseries can deviate from the clearsky model before being considered an anomaly.\
  **night_threshold**: A threshold value for nighttime irradiance.\
  
Returns:\
  **mask**: A pandas Series with boolean values indicating anomalies.\
  **general_threshold**: A pandas Series to plot the boundary of the algorithm

For day values: It identifies an outlier if
```math
\text{Value}(t) > \max \{ \text{clearsky}(t) \times \text{margin}, \text{clearsky}(t) + 50 \}
```
For night values: It identifies as an outlier if:
```math
\text{Value}(t) > 10
```

