from typing import Dict, Iterable, Optional, Union
import numpy as np
from .display import plot_hrir_plane, plot_hrtf_plane, plot_plane_angles, plot_hrir_lines, plot_hrtf_lines
from .full import Cipic, Ari, Listen, BiLi, CrossMod, Ita, Hutubs, Riec, Chedar, Widespread, Sadie2, Princeton3D3A, Scut, Sonicom, MitKemar
from .specifications import HrirPlaneSpec


class PlaneDatasetMixin:
    def __init__(self,
        plane: str,
        domain: str,
        side: str,
        plane_angles: Optional[Iterable[float]],
        plane_offset: float,
        positive_angles: bool,
        distance: Union[float, str],
        hrir_method: Optional[str],
        hrir_variant: str,
        hrir_offset: Optional[float],
        hrir_scaling: Optional[float],
        hrir_samplerate: Optional[float],
        hrir_length: Optional[float],
        hrir_min_phase: bool = False,
        hrir_role: str = 'features',
        other_specs: Optional[Dict] = None,
        **kwargs,
    ):
        hrirs_spec = HrirPlaneSpec(plane, domain, side, plane_angles, plane_offset, positive_angles, distance, hrir_method, hrir_variant, hrir_offset, hrir_scaling, hrir_samplerate, hrir_length, hrir_min_phase)
        if other_specs is None:
            other_specs = {}
        for spec_name in other_specs.keys():
            if hrir_role == spec_name.split('_spec')[0]:
                raise ValueError(f'No {spec_name} should be given since that role is already taken by the HRIRs')
        specs = {hrir_role+'_spec': hrirs_spec, **other_specs}
        super().__init__(**specs, **kwargs)
        if plane in ('horizontal', 'interaural'):
            self.plane_angle_name = 'yaw [°]'
        elif plane in ('median', 'vertical'):
            self.plane_angle_name = 'pitch [°]'
        else: # frontal plane
            self.plane_angle_name = 'roll [°]'


    @property
    def plane_angles(self):
        return self._plane_transform.plane_angles
    

    @property
    def positive_angles(self):
        return self._plane_transform.positive_angles


    @positive_angles.setter
    def positive_angles(self, value):
        self._plane_transform.positive_angles = value


    @property
    def min_angle(self):
        return self._plane_transform.min_angle


    @property
    def max_angle(self):
        return self._plane_transform.max_angle


    def plot_plane(self, idx, ax=None, vmin=None, vmax=None, title=None, lineplot=False, cmap='viridis', continuous=False, colorbar=True, log_freq=False):
        hrir_role = 'features' if 'hrir' in self._features_keys else 'target' if 'hrir' in self._target_keys else 'group'
        if vmin is None or vmax is None:
            all_hrirs = self[:][hrir_role]
            if vmin is None:
                vmin = all_hrirs.min()
            if vmax is None:
                vmax = all_hrirs.max()
        data = self[idx][hrir_role]

        if self._specification['hrir']['domain'] == 'time':
            if lineplot:
                ax = plot_hrir_lines(data, self.plane_angles, self.plane_angle_name, self.hrir_samplerate, ax=ax, vmin=vmin, vmax=vmax)
            else:
                ax = plot_hrir_plane(data, self.plane_angles, self.plane_angle_name, self.hrir_samplerate, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, continuous=continuous, colorbar=colorbar)
        else:
            if lineplot:
                ax = plot_hrtf_lines(data, self.plane_angles, self.plane_angle_name, self.hrtf_frequencies, log_freq=log_freq, ax=ax, vmin=vmin, vmax=vmax)
            else:
                ax = plot_hrtf_plane(data, self.plane_angles, self.plane_angle_name, self.hrtf_frequencies, log_freq=log_freq, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, continuous=continuous, colorbar=colorbar)

        if title is None:
            plane = self._specification['hrir']['plane']
            plane_offset = self._specification['hrir']['plane_offset']
            title = "{} Plane{} of Subject {}'s {} Ear".format(plane.title(),
                ' With Offset {}°'.format(plane_offset) if plane_offset != 0 or plane in ('vertical', 'interaural') else '',
                self.subject_ids[idx],
                self.sides[idx].replace('-', ' ').title(),
            )
        ax.set_title(title)
        return ax


    def plot_angles(self, ax=None, title=None):
        plane = self._specification['hrir']['plane']
        if plane in ('horizontal', 'interaural', 'frontal'):
            zero_location = 'N'
            direction = 'counterclockwise'
        else: # median or vertical
            zero_location = 'W'
            direction = 'clockwise'
        ax = plot_plane_angles(self.plane_angles, self.min_angle, self.max_angle, True, 1, zero_location, direction, ax) # TODO use actual radius
        if title is None:
            plane_offset = self._specification['hrir']['plane_offset']
            title = 'Angles in the {} Plane{}'.format(plane.title(),
                ' With Offset {}°'.format(plane_offset) if plane_offset != 0 or plane in ('vertical', 'interaural') else ''
            )
        ax.set_title(title)
        return ax


class CipicPlane(PlaneDatasetMixin, Cipic):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_offset: Optional[float] = None,
        hrir_scaling: Optional[float] = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        hrir_min_phase: bool = False,
        hrir_role: str = 'features',
        other_specs: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = True,
    ):
        super().__init__(
            plane, domain, side, plane_angles, plane_offset, positive_angles, 'farthest', None, 'compensated', root=root,
            hrir_offset=hrir_offset, hrir_scaling=hrir_scaling, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, hrir_min_phase=hrir_min_phase,
            hrir_role=hrir_role, other_specs=other_specs, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype,
            download=download, verify=verify,
        )


class AriPlane(PlaneDatasetMixin, Ari):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_offset: Optional[float] = None,
        hrir_scaling: Optional[float] = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        hrir_min_phase: bool = False,
        hrir_role: str = 'features',
        other_specs: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = True,
    ):
        if positive_angles is None:
            positive_angles = False
        super().__init__(
            plane, domain, side, plane_angles, plane_offset, positive_angles, 'farthest', None, 'compensated', root=root,
            hrir_offset=hrir_offset, hrir_scaling=hrir_scaling, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, hrir_min_phase=hrir_min_phase,
            hrir_role=hrir_role, other_specs=other_specs, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype,
            download=download, verify=verify,
        )


class ListenPlane(PlaneDatasetMixin, Listen):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_variant: str = 'compensated',
        hrir_offset: Optional[float] = None,
        hrir_scaling: Optional[float] = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        hrir_min_phase: bool = False,
        hrir_role: str = 'features',
        other_specs: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = True,
    ):
        super().__init__(
            plane, domain, side, plane_angles, plane_offset, positive_angles, 'farthest', None, hrir_variant, root=root,
            hrir_offset=hrir_offset, hrir_scaling=hrir_scaling, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, hrir_min_phase=hrir_min_phase,
            hrir_role=hrir_role, other_specs=other_specs, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype,
            download=download, verify=verify,
        )


class CrossModPlane(PlaneDatasetMixin, CrossMod):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_variant: str = 'compensated',
        hrir_offset: Optional[float] = None,
        hrir_scaling: Optional[float] = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        hrir_min_phase: bool = False,
        hrir_role: str = 'features',
        other_specs: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = True,
    ):
        super().__init__(
            plane, domain, side, plane_angles, plane_offset, positive_angles, 'farthest', None, hrir_variant, root=root,
            hrir_offset=hrir_offset, hrir_scaling=hrir_scaling, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, hrir_min_phase=hrir_min_phase,
            hrir_role=hrir_role, other_specs=other_specs, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype,
            download=download, verify=verify,
        )


class BiLiPlane(PlaneDatasetMixin, BiLi):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_variant: str = 'compensated',
        hrir_offset: Optional[float] = None,
        hrir_scaling: Optional[float] = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        hrir_min_phase: bool = False,
        hrir_role: str = 'features',
        other_specs: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = True,
    ):
        super().__init__(
            plane, domain, side, plane_angles, plane_offset, positive_angles, 'farthest', None, hrir_variant, root=root,
            hrir_offset=hrir_offset, hrir_scaling=hrir_scaling, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, hrir_min_phase=hrir_min_phase,
            hrir_role=hrir_role, other_specs=other_specs, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype,
            download=download, verify=verify,
        )


class ItaPlane(PlaneDatasetMixin, Ita):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_offset: Optional[float] = None,
        hrir_scaling: Optional[float] = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        hrir_min_phase: bool = False,
        hrir_role: str = 'features',
        other_specs: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = True,
    ):
        super().__init__(
            plane, domain, side, plane_angles, plane_offset, positive_angles, 'farthest', None, 'compensated', root=root,
            hrir_offset=hrir_offset, hrir_scaling=hrir_scaling, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, hrir_min_phase=hrir_min_phase,
            hrir_role=hrir_role, other_specs=other_specs, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype,
            download=download, verify=verify,
        )


class HutubsPlane(PlaneDatasetMixin, Hutubs):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_method: Optional[str] = 'acoustic',
        hrir_variant: str = 'compensated',
        hrir_offset: Optional[float] = None,
        hrir_scaling: Optional[float] = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        hrir_min_phase: bool = False,
        hrir_role: str = 'features',
        other_specs: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = True,
    ):
        super().__init__(
            plane, domain, side, plane_angles, plane_offset, positive_angles, 'farthest', hrir_method, hrir_variant, root=root,
            hrir_offset=hrir_offset, hrir_scaling=hrir_scaling, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, hrir_min_phase=hrir_min_phase,
            hrir_role=hrir_role, other_specs=other_specs, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype,
            download=download, verify=verify,
        )


class RiecPlane(PlaneDatasetMixin, Riec):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        distance: Union[float, str] = 'farthest',
        hrir_offset: Optional[float] = None,
        hrir_scaling: Optional[float] = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        hrir_min_phase: bool = False,
        hrir_role: str = 'features',
        other_specs: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = True,
    ):
        super().__init__(
            plane, domain, side, plane_angles, plane_offset, positive_angles, distance, None, 'compensated', root=root,
            hrir_offset=hrir_offset, hrir_scaling=hrir_scaling, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, hrir_min_phase=hrir_min_phase,
            hrir_role=hrir_role, other_specs=other_specs, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype,
            download=download, verify=verify,
        )


class ChedarPlane(PlaneDatasetMixin, Chedar):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        distance: Union[float, str] = 'farthest',
        hrir_offset: Optional[float] = None,
        hrir_scaling: Optional[float] = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        hrir_min_phase: bool = False,
        hrir_role: str = 'features',
        other_specs: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ):
        super().__init__(
            plane, domain, side, plane_angles, plane_offset, positive_angles, distance, None, 'compensated', root=root,
            hrir_offset=hrir_offset, hrir_scaling=hrir_scaling, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, hrir_min_phase=hrir_min_phase,
            hrir_role=hrir_role, other_specs=other_specs, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype,
            download=download, verify=verify,
        )


class WidespreadPlane(PlaneDatasetMixin, Widespread):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        distance: Union[float, str] = 'farthest',
        hrir_offset: Optional[float] = None,
        hrir_scaling: Optional[float] = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        hrir_min_phase: bool = False,
        hrir_role: str = 'features',
        other_specs: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        grid: str = 'UV',
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ):
        super().__init__(
            plane, domain, side, plane_angles, plane_offset, positive_angles, distance, None, 'compensated', root=root,
            hrir_offset=hrir_offset, hrir_scaling=hrir_scaling, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, hrir_min_phase=hrir_min_phase,
            hrir_role=hrir_role, other_specs=other_specs, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype, grid=grid,
            download=download, verify=verify,
        )


class Sadie2Plane(PlaneDatasetMixin, Sadie2):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_offset: Optional[float] = None,
        hrir_scaling: Optional[float] = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        hrir_min_phase: bool = False,
        hrir_role: str = 'features',
        other_specs: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = True,
    ):
        super().__init__(
            plane, domain, side, plane_angles, plane_offset, positive_angles, 'farthest', None, 'compensated', root=root,
            hrir_offset=hrir_offset, hrir_scaling=hrir_scaling, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, hrir_min_phase=hrir_min_phase,
            hrir_role=hrir_role, other_specs=other_specs, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype,
            download=download, verify=verify,
        )


class Princeton3D3APlane(PlaneDatasetMixin, Princeton3D3A):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_method: Optional[str] = 'acoustic',
        hrir_variant: str = 'compensated',
        hrir_offset: Optional[float] = None,
        hrir_scaling: Optional[float] = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        hrir_min_phase: bool = False,
        hrir_role: str = 'features',
        other_specs: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = True,
    ):
        super().__init__(
            plane, domain, side, plane_angles, plane_offset, positive_angles, 'farthest', hrir_method, hrir_variant, root=root,
            hrir_offset=hrir_offset, hrir_scaling=hrir_scaling, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, hrir_min_phase=hrir_min_phase,
            hrir_role=hrir_role, other_specs=other_specs, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype,
            download=download, verify=verify,
        )


class ScutPlane(PlaneDatasetMixin, Scut):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        distance: Union[float, str] = 'farthest',
        hrir_method: Optional[str] = 'acoustic',
        hrir_offset: Optional[float] = None,
        hrir_scaling: Optional[float] = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        hrir_min_phase: bool = False,
        hrir_role: str = 'features',
        other_specs: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = True,
    ):
        super().__init__(
            plane, domain, side, plane_angles, plane_offset, positive_angles, distance, hrir_method, 'compensated', root=root,
            hrir_offset=hrir_offset, hrir_scaling=hrir_scaling, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, hrir_min_phase=hrir_min_phase,
            hrir_role=hrir_role, other_specs=other_specs, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype,
            download=download, verify=verify,
        )


class SonicomPlane(PlaneDatasetMixin, Sonicom):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_variant: str = 'compensated',
        hrir_offset: Optional[float] = None,
        hrir_scaling: Optional[float] = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        hrir_min_phase: bool = False,
        hrir_role: str = 'features',
        other_specs: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = True,
    ):
        super().__init__(
            plane, domain, side, plane_angles, plane_offset, positive_angles, 'farthest', None, hrir_variant, root=root,
            hrir_offset=hrir_offset, hrir_scaling=hrir_scaling, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, hrir_min_phase=hrir_min_phase,
            hrir_role=hrir_role, other_specs=other_specs, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype,
            download=download, verify=verify,
        )


class MitKemarPlane(PlaneDatasetMixin, MitKemar):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        distance: Union[float, str] = 'farthest',
        hrir_offset: Optional[float] = None,
        hrir_scaling: Optional[float] = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        hrir_min_phase: bool = False,
        hrir_role: str = 'features',
        other_specs: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = True,
    ):
        super().__init__(
            plane, domain, side, plane_angles, plane_offset, positive_angles, distance, None, 'compensated', root=root,
            hrir_offset=hrir_offset, hrir_scaling=hrir_scaling, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, hrir_min_phase=hrir_min_phase,
            hrir_role=hrir_role, other_specs=other_specs, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype,
            download=download, verify=verify,
        )
