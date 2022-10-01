from typing import Dict, Iterable, Optional
import numpy as np
from .full import CIPIC, ARI, Listen, BiLi, ITA, HUTUBS, RIEC, CHEDAR, Widespread, SADIE2, ThreeDThreeA, SONICOM
from ..display import plot_hrir_plane, plot_hrtf_plane, plot_plane_angles
from ..transforms import PlaneTransform, InterauralPlaneTransform, SphericalPlaneTransform
from ..util import lateral_vertical_from_yaw, lateral_vertical_from_pitch, lateral_vertical_from_roll, azimuth_elevation_from_yaw, azimuth_elevation_from_pitch, azimuth_elevation_from_roll


class PlaneMixin:
    def __init__(self,
        plane: str,
        domain: str,
        side: str,
        plane_offset: float,
        row_angles: Iterable[float],
        column_angles: Iterable[float],
        planar_transform: PlaneTransform,
        hrir_samplerate: Optional[float],
        hrir_length: Optional[float],
        **kwargs,
    ):
        if plane not in ('horizontal', 'median', 'frontal', 'vertical', 'interaural'):
            raise ValueError('Unknown plane "{}", needs to be "horizontal", "median", "frontal", "vertical" or "interaural".')
        self._plane = plane
        self._plane_offset = plane_offset
        self._domain = domain
        feature_spec = {'hrirs': {'row_angles': row_angles, 'column_angles': column_angles, 'side': side, 'domain': domain, 'samplerate': hrir_samplerate, 'length': hrir_length}}
        super().__init__(feature_spec=feature_spec, hrir_transform=planar_transform, **kwargs)
        self.plane_angles = self._hrir_transform.calc_plane_angles(self.row_angles, self.column_angles, self._selection_mask)


    @property
    def positive_angles(self):
        return self._hrir_transform.positive_angles


    @positive_angles.setter
    def positive_angles(self, value):
        self._hrir_transform.positive_angles = value
        self.plane_angles = self._hrir_transform.calc_plane_angles(self.row_angles, self.column_angles, self._selection_mask)


    @property
    def min_angle(self):
        return self._hrir_transform.min_angle


    @property
    def max_angle(self):
        return self._hrir_transform.max_angle


    def plot_plane(self, idx, ax=None, cmap='viridis', continuous=False, vmin=None, vmax=None, title=None, colorbar=True, log_freq=False):
        if self._plane in ('horizontal', 'interaural'):
            angles_label = 'yaw [°]'
        elif self._plane in ('median', 'vertical'):
            angles_label = 'pitch [°]'
        else: # frontal plane
            angles_label = 'roll [°]'
        if vmin is None or vmax is None:
            all_features = self[:]['features']
            if vmin is None:
                vmin = all_features.min()
            if vmax is None:
                vmax = all_features.max()
        item = self[idx]
        data = item['features'].T

        if self._domain == 'time':
            ax = plot_hrir_plane(data, self.plane_angles, angles_label, self.hrir_samplerate, ax=ax, cmap=cmap, continuous=continuous, vmin=vmin, vmax=vmax, colorbar=colorbar)
        else:
            ax = plot_hrtf_plane(data, self.plane_angles, angles_label, self.hrtf_frequencies, log_freq=log_freq, ax=ax, cmap=cmap, continuous=continuous, vmin=vmin, vmax=vmax, colorbar=colorbar)

        if title is None:
            title = "{} Plane{} of Subject {}'s {} Ear".format(self._plane.title(),
                ' With Offset {}°'.format(self._plane_offset) if self._plane_offset != 0 or self._plane in ('vertical', 'interaural') else '',
                self.subject_ids[idx],
                self.sides[idx].replace('-', ' ').title(),
            )
        ax.set_title(title)
        return ax


    def plot_angles(self, ax=None, title=None):
        if self._plane in ('horizontal', 'interaural', 'frontal'):
            zero_location = 'N'
            direction = 'counterclockwise'
        else: # median or vertical
            zero_location = 'W'
            direction = 'clockwise'
        ax = plot_plane_angles(self.plane_angles, self.min_angle, self.max_angle, True, 1, zero_location, direction, ax) # TODO use actual radius
        if title is None:
            title = 'Angles in the {} Plane{}'.format(self._plane.title(),
                ' With Offset {}°'.format(self._plane_offset) if self._plane_offset != 0 or self._plane in ('vertical', 'interaural') else ''
            )
        ax.set_title(title)
        return ax


class InterauralPlaneMixin(PlaneMixin):
    def __init__(self,
        plane: str,
        domain: str,
        side: str,
        plane_angles: Optional[Iterable[float]],
        plane_offset: float,
        positive_angles: bool,
        hrir_samplerate: Optional[float],
        hrir_length: Optional[float],
        **kwargs,
    ):
        if plane == 'horizontal':
            if plane_offset != 0:
                raise ValueError('Only the horizontal plane at vertical angle 0 is available in an interaural coordinate dataset')
            lateral_angles, vertical_angles = InterauralPlaneMixin._lateral_vertical_from_yaw(plane_angles, plane_offset)
        elif plane == 'median':
            lateral_angles, vertical_angles = InterauralPlaneMixin._lateral_vertical_from_pitch(plane_angles, plane_offset)
        elif plane == 'frontal':
            if plane_offset != 0:
                raise ValueError('Only the frontal plane at vertical angles +/-90 is available in an interaural coordinate dataset')
            lateral_angles, vertical_angles = InterauralPlaneMixin._lateral_vertical_from_roll(plane_angles, plane_offset)
        elif plane == 'interaural':
            lateral_angles, vertical_angles = InterauralPlaneMixin._lateral_vertical_from_yaw(plane_angles, plane_offset)
        else:
            if plane not in ('horizontal', 'median', 'frontal', 'interaural'):
                raise ValueError('Unknown plane "{}", needs to be "horizontal", "median", "frontal" or "interaural".')

        plane_transform = InterauralPlaneTransform(plane, plane_offset, positive_angles)
        super().__init__(plane, domain, side, plane_offset, vertical_angles, lateral_angles, plane_transform, hrir_samplerate, hrir_length, **kwargs)


    @staticmethod
    def _lateral_vertical_from_yaw(yaw_angles, plane_offset=0):
        if yaw_angles is None:
            return None, (plane_offset - 180, plane_offset)
        return lateral_vertical_from_yaw(yaw_angles, plane_offset)


    @staticmethod
    def _lateral_vertical_from_pitch(pitch_angles, plane_offset=0):
        if pitch_angles is None:
            return (plane_offset,), None
        return lateral_vertical_from_pitch(pitch_angles, plane_offset)


    @staticmethod
    def _lateral_vertical_from_roll(roll_angles, plane_offset=0):
        if roll_angles is None:
            return None, (plane_offset - 90, plane_offset + 90)
        return lateral_vertical_from_roll(roll_angles, plane_offset)


class SphericalPlaneMixin(PlaneMixin):
    def __init__(self,
        plane: str,
        domain: str,
        side: str,
        plane_angles: Optional[Iterable[float]],
        plane_offset: float,
        positive_angles: bool,
        hrir_samplerate: Optional[float],
        hrir_length: Optional[float],
        **kwargs,
    ):
        if plane == 'horizontal':
            azimuth_angles, elevation_angles = SphericalPlaneMixin._azimuth_elevation_from_yaw(plane_angles, plane_offset)
        elif plane == 'median':
            if plane_offset != 0:
                raise ValueError('Only the median plane at azimuth 0 is available in a spherical coordinate dataset')
            azimuth_angles, elevation_angles = SphericalPlaneMixin._azimuth_elevation_from_pitch(plane_angles, plane_offset)
        elif plane == 'frontal':
            if plane_offset != 0:
                raise ValueError('Only the frontal plane at azimuth +/-90 is available in a spherical coordinate dataset')
            azimuth_angles, elevation_angles = SphericalPlaneMixin._azimuth_elevation_from_roll(plane_angles, plane_offset)
        elif plane == 'vertical':
            azimuth_angles, elevation_angles = SphericalPlaneMixin._azimuth_elevation_from_pitch(plane_angles, plane_offset)
        else:
            raise ValueError('Unknown plane "{}", needs to be "horizontal", "median", "frontal" or "vertical".')

        plane_transform = SphericalPlaneTransform(plane, plane_offset, positive_angles)
        super().__init__(plane, domain, side, plane_offset, azimuth_angles, elevation_angles, plane_transform, hrir_samplerate, hrir_length, **kwargs)


    @staticmethod
    def _azimuth_elevation_from_yaw(yaw_angles, plane_offset=0):
        if yaw_angles is None:
            return None, (plane_offset,)
        return azimuth_elevation_from_yaw(yaw_angles, plane_offset)


    @staticmethod
    def _azimuth_elevation_from_pitch(pitch_angles, plane_offset=0):
        if pitch_angles is None:
            return (plane_offset - 180, plane_offset), None
        return azimuth_elevation_from_pitch(pitch_angles, plane_offset)


    @staticmethod
    def _azimuth_elevation_from_roll(roll_angles, plane_offset=0):
        if roll_angles is None:
            return (plane_offset - 90, plane_offset + 90), None
        return azimuth_elevation_from_roll(roll_angles, plane_offset)


class CIPICPlane(InterauralPlaneMixin, CIPIC):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'both-left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        dtype: type = np.float32,
    ):
        super().__init__(plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length,
            root=root, target_spec=target_spec, group_spec=group_spec, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype)


class ARIPlane(SphericalPlaneMixin, ARI):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'both-left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        dtype: type = np.float32,
    ):
        if positive_angles is None:
            positive_angles = False
        super().__init__(plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length,
            root=root, target_spec=target_spec, group_spec=group_spec, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype)


class ListenPlane(SphericalPlaneMixin, Listen):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'both-left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        hrtf_type: str = 'compensated',
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        dtype: type = np.float32,
    ):
        super().__init__(plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length,
            root=root, target_spec=target_spec, group_spec=group_spec, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype, hrtf_type=hrtf_type)


class BiLiPlane(SphericalPlaneMixin, BiLi):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'both-left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        hrtf_type: str = 'compensated',
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        dtype: type = np.float32,
    ):
        super().__init__(plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length,
            root=root, target_spec=target_spec, group_spec=group_spec, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype, hrtf_type=hrtf_type)


class ITAPlane(SphericalPlaneMixin, ITA):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'both-left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        dtype: type = np.float32,
    ):
        super().__init__(plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length,
            root=root, target_spec=target_spec, group_spec=group_spec, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype)


class HUTUBSPlane(SphericalPlaneMixin, HUTUBS):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'both-left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        measured_hrtf: bool = True,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        dtype: type = np.float32,
    ):
        super().__init__(plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length,
            root=root, target_spec=target_spec, group_spec=group_spec, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype, measured_hrtf=measured_hrtf)


class RIECPlane(SphericalPlaneMixin, RIEC):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'both-left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        dtype: type = np.float32,
    ):
        super().__init__(plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length,
            root=root, target_spec=target_spec, group_spec=group_spec, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype)


class CHEDARPlane(SphericalPlaneMixin, CHEDAR):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'both-left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        radius: float = 1,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        dtype: type = np.float32,
    ):
        super().__init__(plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length,
            root=root, target_spec=target_spec, group_spec=group_spec, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype, radius=radius)


class WidespreadPlane(SphericalPlaneMixin, Widespread):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'both-left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        radius: float = 1,
        grid: str = 'UV',
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        dtype: type = np.float32,
    ):
        super().__init__(plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length,
            root=root, target_spec=target_spec, group_spec=group_spec, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype, radius=radius, grid=grid)


class SADIE2Plane(SphericalPlaneMixin, SADIE2):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'both-left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        dtype: type = np.float32,
    ):
        super().__init__(plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length,
            root=root, target_spec=target_spec, group_spec=group_spec, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype)


class ThreeDThreeAPlane(SphericalPlaneMixin, ThreeDThreeA):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'both-left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        hrtf_method: str = 'measured',
        hrtf_type: str = 'compensated',
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        dtype: type = np.float32,
    ):
        super().__init__(plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length,
            root=root, target_spec=target_spec, group_spec=group_spec, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype, hrtf_method=hrtf_method, hrtf_type=hrtf_type)


class SONICOMPlane(SphericalPlaneMixin, SONICOM):
    def __init__(self,
        root: str,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'both-left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        hrtf_type: str = 'compensated',
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        dtype: type = np.float32,
    ):
        super().__init__(plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length,
            root=root, target_spec=target_spec, group_spec=group_spec, subject_ids=subject_ids, exclude_ids=exclude_ids, dtype=dtype, hrtf_type=hrtf_type)
