import matplotlib.pyplot as plt
import numpy as np


def plot_hrtf_plane(hrtf, angles, angles_label, frequencies, log_freq=False, ax=None, vmin=None, vmax=None, cmap='gray', continuous=False, colorbar=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    mesh = ax.pcolormesh(angles, frequencies/1000, hrtf.squeeze().T, shading='gouraud' if continuous else 'nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        fig.colorbar(mesh, ax=ax)
    ax.set_xlabel(angles_label)
    if log_freq:
        ax.set_yscale('log')
        ax.set_ylim([frequencies[1]/1000, frequencies[-1]/1000])
    ax.set_ylabel('frequency [kHz]')
    return ax


def plot_hrir_plane(hrir, angles, angles_label, sample_rate, ax=None, vmin=None, vmax=None, cmap='gray', continuous=False, colorbar=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    times = np.arange(0, hrir.shape[-1]*1000/sample_rate, 1000/sample_rate)
    mesh = ax.pcolormesh(angles, times, hrir.squeeze().T, shading='gouraud' if continuous else 'nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        fig.colorbar(mesh, ax=ax)
    ax.set_xlabel(angles_label)
    ax.set_ylabel('time [ms]')
    return ax


def plot_plane_positions(angles, min_angle, max_angle, closed_open_angles, radii, zero_location, direction, ax=None, radius_limit=None, **plot_kwargs):
    if ax is None:
        _, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    plot_kwargs = dict(color='k', marker='o', linestyle='') | plot_kwargs
    for radius in radii:
        ax.plot(np.deg2rad(angles), np.full(len(angles), radius), **plot_kwargs)
    if radius_limit is None:
        radius_limit = radii.max() * 1.2
    ax.set_rmax(radius_limit)
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


def plot_3d_positions(cartesian_positions, ax=None, ax_limit=None, **scatter_kwargs):
    cartesian_positions = np.asanyarray(cartesian_positions)
    if ax is None:
        _, ax = plt.subplots(subplot_kw={'projection': '3d'}, layout='constrained')
    elif 'Axes3D' not in str(ax):
        raise ValueError('Three-dimensional axes are required for plotting positions.')
    scatter_kwargs = dict(marker='.') | scatter_kwargs
    ax.scatter(*cartesian_positions.T, **scatter_kwargs)
    ax.azim = -45
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    if ax_limit is None:
        ax_limit = np.abs(cartesian_positions).max()
    ax.set_xlim(-ax_limit, ax_limit)
    ax.set_ylim(-ax_limit, ax_limit)
    ax.set_zlim(-ax_limit, ax_limit)
    return ax


def plot_hrtf_lines(hrtf, angle_labels, angles_title, frequencies, log_freq=False, ax=None, vmin=None, vmax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(frequencies/1000, hrtf.squeeze().T, label=angle_labels)
    ax.set_xlabel('frequency [kHz]')
    if log_freq:
        ax.set_xscale('log')
        ax.set_xlim([frequencies[1]/1000, frequencies[-1]/1000])
    ax.set_ylim([vmin, vmax])
    ax.legend(title=angles_title, loc='upper right')
    return ax


def plot_hrir_lines(hrir, angle_labels, angles_title, sample_rate, ax=None, vmin=None, vmax=None):
    if ax is None:
        _, ax = plt.subplots()
    times = np.arange(0, hrir.shape[-1]*1000/sample_rate, 1000/sample_rate)
    ax.plot(times, hrir.squeeze().T, label=angle_labels)
    ax.set_xlabel('time [ms]')
    ax.set_ylim([vmin, vmax])
    ax.legend(title=angles_title, loc='upper right')
    return ax
