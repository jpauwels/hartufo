import matplotlib.pyplot as plt
import numpy as np


def plot_hrtf_plane(hrtf, angles, angles_label, frequencies, log_freq=False, ax=None, cmap='gray', continuous=False, vmin=None, vmax=None, colorbar=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    mesh = ax.pcolormesh(angles, frequencies/1000, hrtf, shading='gouraud' if continuous else 'nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        fig.colorbar(mesh, ax=ax)
    ax.set_xlabel(angles_label)
    if log_freq:
        ax.set_yscale('log')
        ax.set_ylim([frequencies[1]/1000, frequencies[-1]/1000])
    ax.set_ylabel('frequency [kHz]')
    return ax


def plot_hrir_plane(hrir, angles, angles_label, sample_rate, ax=None, cmap='gray', continuous=False, vmin=None, vmax=None, colorbar=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    times = np.arange(0, hrir.shape[0]*1000/sample_rate, 1000/sample_rate)
    mesh = ax.pcolormesh(angles, times, hrir, shading='gouraud' if continuous else 'nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        fig.colorbar(mesh, ax=ax)
    ax.set_xlabel(angles_label)
    ax.set_ylabel('time [ms]')
    return ax


def plot_plane_angles(angles, min_angle, max_angle, closed_open_angles, radius, zero_location, direction, ax=None):
    if ax is None:
        _, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(np.deg2rad(angles), np.full(len(angles), radius), 'ko')
    ax.set_rmax(radius * 1.2)
    ax.set_rticks([]) # no radial ticks
    ax.grid(False)
    if closed_open_angles:
        angular_ticks = np.linspace(min_angle, max_angle, 8, endpoint=False)
    else:
        angular_ticks = np.flip(np.linspace(max_angle, min_angle, 8, endpoint=False))
    ax.set_xticks(np.deg2rad(angular_ticks))
    ax.set_thetamin(min_angle)
    ax.set_thetamax(max_angle)
    ax.set_theta_direction(direction)
    ax.set_theta_zero_location(zero_location)
    return ax
