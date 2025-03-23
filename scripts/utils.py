import pvlib
import pandas as pd

def validate_pvlib_location(location: pvlib.location.Location):
    """
    Checks the input of a function.
    
    Args:
        location: The pvlib location object to check.

    Raises:
        TypeError: if the input isn't a pvlib.location.Location object.
    """
    if not isinstance(location, pvlib.location.Location):
        raise TypeError("location must be a pvlib.location.Location object.")

def validate_timezone_aware(times: pd.DatetimeIndex):
    """
    Checks the input of a function.
    Args:
        times: the DatetimeIndex of a series to check
    Raises:
        TypeError: If the input isn't a pandas.DatetimeIndex
        ValueError: If the input isn't timezone-aware.
    """
    if not isinstance(times, pd.DatetimeIndex):
        raise TypeError("Input must be a pandas.DatetimeIndex")
    if times.tz is None:
        raise ValueError("DatetimeIndex must be timezone-aware.")
    
if __name__ == '__main__':
    print(help(validate_pvlib_location))
    print(help(validate_timezone_aware))
