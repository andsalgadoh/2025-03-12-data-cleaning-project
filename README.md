# WORK IN PROGRESS

This is part of my personal Python practice.

The code here is meant to:
1) Show how a certain algorithm can be used to identify and remove incorrect values, caused by sensor malfunction, from a dataset.
2) Evaluate the impact of data cleaning on time series forecast accuracy using simple neural network models.

My goals for this project are:
1) Get familiarized with Python and its libraries: numpy, pandas, matplotlib, pytorch.
2) Build interactive visualizations for my data-cleaning algorithms. I'm specially interested in viewing the effect of tuning a certain parameter in real time, to see what data points are chosen for each tolerance value.

## Scripts Folder:
### anomaly_detection.py
function: anomaly_clearsky()\
Detects outliers based on a threshold using a clearsky-irradiance-model and a margin input

**Inputs:**\
timeseries\
location\
irradiance_type

Optional:\
margin (default = 1.2)\
max_night_irradiance (default = 10)
  
Implements a threshold using the function:
```math
\text{Threshold}(t) = \max \{ \text{clearsky}(t) \times \text{margin}, \text{clearsky}(t) + 50 \}
```

Returns:\
**mask** - boolean array mean to be used to retrieve only valid values from the timeseries\
**irradiance_threshold** - The resulting timeseries that serves as the threshold.
