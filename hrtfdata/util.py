import numpy as np

def wrap_closed_open_interval(values, lower, upper):
    return (np.asarray(values) - lower) % (upper - lower) + lower


def wrap_open_closed_interval(values, lower, upper):
    return -((lower - np.asarray(values)) % (upper - lower)) - lower


def spherical2interaural(azimuth, elevation, radius, angles_in_degrees=True):
    if angles_in_degrees:
        azimuth = np.deg2rad(azimuth)
        elevation = np.deg2rad(elevation)
    lateral = np.arcsin(np.sin(azimuth) * np.cos(elevation))
    vertical = np.arctan2(np.tan(elevation), np.cos(azimuth))
    if angles_in_degrees:
        return np.rad2deg(lateral), np.rad2deg(vertical), radius
    else:
        return lateral, vertical, radius


def spherical2cartesian(azimuth, elevation, radius, angles_in_degrees=True):
    if angles_in_degrees:
        azimuth = np.deg2rad(azimuth)
        elevation = np.deg2rad(elevation)
    hoz_proj = radius * np.cos(elevation)
    x = hoz_proj * np.cos(azimuth)
    y = hoz_proj * np.sin(azimuth)
    z = radius * np.sin(elevation)
    return x, y, z


def cartesian2spherical(x, y, z, angles_in_degrees=True):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    radius = np.sqrt(x**2 + y**2 + z**2)
    if angles_in_degrees:
        return np.rad2deg(azimuth), np.rad2deg(elevation), radius
    else:
        return azimuth, elevation, radius


def cartesian2interaural(x, y, z, angles_in_degrees=True):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    lateral = np.arctan2(y, np.sqrt(x**2 + z**2))
    vertical = np.arctan2(z, x) 
    radius = np.sqrt(x**2 + y**2 + z**2)
    if angles_in_degrees:
        return np.rad2deg(lateral), np.rad2deg(vertical), radius
    else:
        return lateral, vertical, radius


def interaural2spherical(lateral, vertical, radius, angles_in_degrees=True):
    if angles_in_degrees:
        lateral = np.deg2rad(lateral)
        vertical = np.deg2rad(vertical)
    azimuth = np.arctan2(np.tan(lateral), np.cos(vertical))
    elevation = np.arcsin(np.cos(lateral) * np.sin(vertical))
    if angles_in_degrees:
        return np.rad2deg(azimuth), np.rad2deg(elevation), radius
    else:
        return azimuth, elevation, radius


def interaural2cartesian(lateral, vertical, radius, angles_in_degrees=True):
    if angles_in_degrees:
        lateral = np.deg2rad(lateral)
        vertical = np.deg2rad(vertical)
    med_proj = radius * np.cos(lateral)
    x = med_proj * np.cos(vertical)
    y = radius * np.sin(lateral)
    z = med_proj * np.sin(vertical)
    return x, y, z
