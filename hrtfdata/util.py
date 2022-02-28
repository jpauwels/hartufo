import numpy as np

def wrap_closed_open_interval(values, lower, upper):
    return (values - lower) % (upper - lower) + lower


def wrap_open_closed_interval(values, lower, upper):
    return -((lower - values) % (upper - lower)) - lower


def spherical2interaural(azimuth, elevation, radius, angles_in_degrees=True):
    if angles_in_degrees:
        azimuth = np.deg2rad(azimuth)
        elevation = np.deg2rad(elevation)
    lateral = np.arcsin(np.sin(azimuth) * np.cos(elevation))
    vertical = np.arctan(np.tan(elevation) / np.cos(azimuth))
    vertical[np.cos(azimuth) < 0] += np.pi
    if angles_in_degrees:
        return np.rad2deg(lateral), np.rad2deg(vertical), radius
    else:
        return lateral, vertical, radius


def spherical2cartesian(azimuth, elevation, radius, angles_in_degrees=True):
    if angles_in_degrees:
        azimuth = np.deg2rad(azimuth)
        elevation = np.deg2rad(elevation)
    x = np.cos(azimuth)
    y = np.sin(azimuth)
    z = np.sin(elevation)
    return x, y, z
