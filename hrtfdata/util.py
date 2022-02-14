import numpy as np

def wrap_closed_open_interval(values, lower, upper):
    return (values - lower) % (upper - lower) + lower


def wrap_open_closed_interval(values, lower, upper):
    return -((lower - values) % (upper - lower)) - lower


def spherical2interaural(azimuth, elevation, radius, angle_unit='degrees'):
    if angle_unit == 'degrees':
        azimuth = np.deg2rad(azimuth)
        elevation = np.deg2rad(elevation)
    elif angle_unit != 'radians':
        raise ValueError(f'Unknown angle unit "{angle_unit}"')
    lateral = np.arcsin(np.sin(azimuth) * np.cos(elevation))
    vertical = np.arctan(np.tan(elevation) / np.cos(azimuth))
    vertical[np.cos(azimuth) < 0] += np.pi
    if angle_unit == 'degrees':
        return np.rad2deg(lateral), np.rad2deg(vertical), radius
    else:
        return lateral, vertical, radius


def spherical2cartesian(azimuth, elevation, radius, angle_unit='degrees'):
    if angle_unit == 'degrees':
        azimuth = np.deg2rad(azimuth)
        elevation = np.deg2rad(elevation)
    elif angle_unit != 'radians':
        raise ValueError(f'Unknown angle unit "{angle_unit}"')
    x = np.cos(azimuth)
    y = np.sin(azimuth)
    z = np.sin(elevation)
    if angle_unit == 'degrees':
        return np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    else:
        return x, y, z
