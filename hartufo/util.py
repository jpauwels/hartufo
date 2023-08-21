import numpy as np
from typing import Iterable


def wrap_closed_open_interval(values, lower, upper):
    return (np.asarray(values) - lower) % (upper - lower) + lower


def wrap_open_closed_interval(values, lower, upper):
    return -((lower - np.asarray(values)) % (upper - lower)) - lower


def wrap_closed_interval(values, lower, upper):
    values = np.asarray(values)
    return np.where((values < lower) | (values > upper), wrap_closed_open_interval(values, lower, upper), values)


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


def azimuth_elevation_from_yaw(yaw_angles, plane_offset=0):
    norm_yaw = wrap_closed_open_interval(yaw_angles, -180, 180)
    try:
        return tuple(norm_yaw), (plane_offset,) * len(norm_yaw)
    except TypeError:
        return (norm_yaw,), (plane_offset,)


def azimuth_elevation_from_pitch(pitch_angles, plane_offset=0):
    norm_pitch = wrap_closed_open_interval(pitch_angles, -90, 270)
    azimuth_angles = np.where(norm_pitch < 90, plane_offset, plane_offset - 180)
    elevation_angles = np.where(norm_pitch < 90, norm_pitch, 180 - norm_pitch)
    try:
        return tuple(azimuth_angles), tuple(elevation_angles)
    except TypeError:
        return (azimuth_angles.item(),), (elevation_angles.item(),)


def azimuth_elevation_from_roll(roll_angles, plane_offset=0):
    norm_roll = wrap_closed_open_interval(roll_angles, -180, 180)
    azimuth_angles = np.where(norm_roll < 0, plane_offset + 90, plane_offset - 90)
    elevation_angles = np.where(norm_roll < 0, norm_roll + 90, 90 - norm_roll)
    try:
        return tuple(azimuth_angles), tuple(elevation_angles)
    except TypeError:
        return (azimuth_angles.item(),), (elevation_angles.item(),)


def lateral_vertical_from_yaw(yaw_angles, plane_offset=0):
    norm_yaw = wrap_closed_open_interval(yaw_angles, -90, 270)
    lateral_angles = np.where(norm_yaw < 90, norm_yaw, 180 - norm_yaw)
    vertical_angles = np.where(norm_yaw < 90, plane_offset, plane_offset - 180)
    try:
        return tuple(lateral_angles), tuple(vertical_angles)
    except TypeError:
        return (lateral_angles.item(),), (vertical_angles.item(),)


def lateral_vertical_from_pitch(pitch_angles, plane_offset=0):
    norm_pitch = wrap_closed_open_interval(pitch_angles, -180, 180)
    try:
        return (plane_offset,) * len(norm_pitch), tuple(norm_pitch)
    except TypeError:
        return (plane_offset,), (norm_pitch,)


def lateral_vertical_from_roll(roll_angles, plane_offset=0):
    norm_roll = wrap_closed_open_interval(roll_angles, -90, 270)
    lateral_angles = np.where(norm_roll < 90, -norm_roll, norm_roll - 180)
    vertical_angles = np.where(norm_roll < 90, plane_offset + 90, plane_offset - 90)
    try:
        return tuple(lateral_angles), tuple(vertical_angles)
    except TypeError:
        return (lateral_angles.item(),), (vertical_angles.item(),)


def quantise(values, precision):
    if precision > 0:
        return (values / precision).round(decimals=0) * precision
    else:
        return values.round(decimals=-precision)
