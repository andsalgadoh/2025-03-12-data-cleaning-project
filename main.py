import matplotlib.pyplot as plt
import numpy
import pandas


from utils import detect_anomalies

# Example dataset
data = [25, 26, 1000, 27, 28, -500, 29]
anomalies = detect_anomalies(data, tol=1.0)

print(anomalies)  # Boolean mask for anomalies

# THESE CHANGES WERE MADE ON GITHUB WEBSITE
# aaaaaaaaaaaaaaaaa
# bbbbbbbbbbbbbbbbb
