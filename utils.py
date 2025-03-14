import numpy as np

def detect_anomalies(data, tol=3.0):
    """
    Detects anomalies based on a threshold (tol) for standard deviation.
    Returns a mask of anomalies.
    """
    mean = np.mean(data)
    std = np.std(data)
    return (data < mean - tol * std) | (data > mean + tol * std)

