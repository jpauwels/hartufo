from pathlib import Path
from typing import Dict, Iterable, Optional
import numpy as np
from .full import HRTFDataset
from ..datapoint import DataPoint, CipicDataPoint, AriDataPoint, ListenDataPoint, BiLiDataPoint, ItaDataPoint, HutubsDataPoint, RiecDataPoint, ChedarDataPoint, WidespreadDataPoint, Sadie2DataPoint, ThreeDThreeADataPoint, SonicomDataPoint
from ..display import plot_hrir_plane, plot_hrtf_plane, plot_plane_angles
from ..transforms import PlaneTransform, InterauralPlaneTransform, SphericalPlaneTransform
from ..util import wrap_closed_open_interval


class HRTFPlaneDataset(HRTFDataset):
    def __init__(self,
        datapoint: DataPoint,
        plane: str,
        domain: str,
        side: str,
        plane_offset: float,
        row_angles: Iterable[float],
        column_angles: Iterable[float],
        planar_transform: PlaneTransform,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[float] = None,
        subject_ids: Optional[Iterable[int]] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
    ):
        self._plane = plane
        self._plane_offset = plane_offset
        if plane not in ('horizontal', 'median', 'frontal', 'vertical', 'interaural'):
            raise ValueError('Unknown plane "{}", needs to be "horizontal", "median", "frontal", "vertical" or "interaural".')
        self._domain = domain
        self._planar_transform = planar_transform

        feature_spec = {'hrirs': {'row_angles': row_angles, 'column_angles': column_angles, 'side': side, 'domain': domain, 'samplerate': hrir_samplerate, 'length': hrir_length}}
        super().__init__(datapoint, feature_spec, target_spec, group_spec, subject_ids, None, exclude_ids, hrir_transform=planar_transform)
        self.positive_angles = planar_transform.positive_angles


    @property
    def positive_angles(self):
        return self._planar_transform.positive_angles


    @positive_angles.setter
    def positive_angles(self, value):
        self._planar_transform.positive_angles = value
        self.plane_angles = self._planar_transform.calc_plane_angles(self.row_angles, self.column_angles, self._selection_mask)
        self.min_angle = self._planar_transform.min_angle
        self.max_angle = self._planar_transform.max_angle


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


class InterauralPlaneDataset(HRTFPlaneDataset):
    def __init__(self,
        datapoint: DataPoint,
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
    ):
        if plane == 'horizontal':
            if plane_offset != 0:
                raise ValueError('Only the horizontal plane at vertical angle 0 is available in an interaural coordinate dataset')
            lateral_angles, vertical_angles = InterauralPlaneDataset._lateral_vertical_from_yaw(plane_angles, plane_offset)
        elif plane == 'median':
            lateral_angles, vertical_angles = InterauralPlaneDataset._lateral_vertical_from_pitch(plane_angles, plane_offset)
        elif plane == 'frontal':
            if plane_offset != 0:
                raise ValueError('Only the frontal plane at vertical angles +/-90 is available in an interaural coordinate dataset')
            lateral_angles, vertical_angles = InterauralPlaneDataset._lateral_vertical_from_roll(plane_angles, plane_offset)
        elif plane == 'interaural':
            lateral_angles, vertical_angles = InterauralPlaneDataset._lateral_vertical_from_yaw(plane_angles, plane_offset)
        else:
            if plane not in ('horizontal', 'median', 'frontal', 'interaural'):
                raise ValueError('Unknown plane "{}", needs to be "horizontal", "median", "frontal" or "interaural".')

        plane_transform = InterauralPlaneTransform(plane, plane_offset, positive_angles)
        super().__init__(datapoint, plane, domain, side, plane_offset, vertical_angles, lateral_angles, plane_transform, hrir_samplerate, hrir_length, subject_ids, exclude_ids, target_spec, group_spec)


    @staticmethod
    def _lateral_vertical_from_yaw(yaw_angles, plane_offset=0):
        if yaw_angles is None:
            return None, (plane_offset - 180, plane_offset)
        norm_yaw = wrap_closed_open_interval(yaw_angles, -90, 270)
        lateral_angles = tuple(np.where(norm_yaw < 90, norm_yaw, 180 - norm_yaw))
        vertical_angles = tuple(np.where(norm_yaw < 90, plane_offset, plane_offset - 180))
        return lateral_angles, vertical_angles


    @staticmethod
    def _lateral_vertical_from_pitch(pitch_angles, plane_offset=0):
        if pitch_angles is None:
            return (plane_offset,), None
        norm_pitch = wrap_closed_open_interval(pitch_angles, -180, 180)
        if not isinstance(norm_pitch, Iterable):
            norm_pitch = (norm_pitch,)
        return (plane_offset,) * len(norm_pitch), tuple(norm_pitch)


    @staticmethod
    def _lateral_vertical_from_roll(roll_angles, plane_offset=0):
        if roll_angles is None:
            return None, (plane_offset - 90, plane_offset + 90)
        norm_roll = wrap_closed_open_interval(roll_angles, -90, 270)
        lateral_angles = tuple(np.where(norm_roll < 90, -norm_roll, norm_roll - 180))
        vertical_angles = tuple(np.where(norm_roll < 90, plane_offset + 90, plane_offset - 90))
        return lateral_angles, vertical_angles


class SphericalPlaneDataset(HRTFPlaneDataset):
    def __init__(self,
        datapoint: DataPoint,
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
    ):
        if plane == 'horizontal':
            azimuth_angles, elevation_angles = SphericalPlaneDataset._azimuth_elevation_from_yaw(plane_angles, plane_offset)
        elif plane == 'median':
            if plane_offset != 0:
                raise ValueError('Only the median plane at azimuth 0 is available in a spherical coordinate dataset')
            azimuth_angles, elevation_angles = SphericalPlaneDataset._azimuth_elevation_from_pitch(plane_angles, plane_offset)
        elif plane == 'frontal':
            if plane_offset != 0:
                raise ValueError('Only the frontal plane at azimuth +/-90 is available in a spherical coordinate dataset')
            azimuth_angles, elevation_angles = SphericalPlaneDataset._azimuth_elevation_from_roll(plane_angles, plane_offset)
        elif plane == 'vertical':
            azimuth_angles, elevation_angles = SphericalPlaneDataset._azimuth_elevation_from_pitch(plane_angles, plane_offset)
        else:
            raise ValueError('Unknown plane "{}", needs to be "horizontal", "median", "frontal" or "vertical".')

        plane_transform = SphericalPlaneTransform(plane, plane_offset, positive_angles)
        super().__init__(datapoint, plane, domain, side, plane_offset, azimuth_angles, elevation_angles, plane_transform, hrir_samplerate, hrir_length, subject_ids, exclude_ids, target_spec, group_spec)


    @staticmethod
    def _azimuth_elevation_from_yaw(yaw_angles, plane_offset=0):
        if yaw_angles is None:
            return None, (plane_offset,)
        norm_yaw = wrap_closed_open_interval(yaw_angles, -180, 180)
        if not isinstance(norm_yaw, Iterable):
            norm_yaw = (norm_yaw,)
        return tuple(norm_yaw), (plane_offset,) * len(norm_yaw)


    @staticmethod
    def _azimuth_elevation_from_pitch(pitch_angles, plane_offset=0):
        if pitch_angles is None:
            return (plane_offset - 180, plane_offset), None
        norm_pitch = wrap_closed_open_interval(pitch_angles, -90, 270)
        azimuth_angles = tuple(np.where(norm_pitch < 90, plane_offset, plane_offset - 180))
        elevation_angles = tuple(np.where(norm_pitch < 90, norm_pitch, 180 - norm_pitch))
        return azimuth_angles, elevation_angles


    @staticmethod
    def _azimuth_elevation_from_roll(roll_angles, plane_offset=0):
        if roll_angles is None:
            return (plane_offset - 90, plane_offset + 90), None
        norm_roll = wrap_closed_open_interval(roll_angles, -180, 180)
        azimuth_angles = tuple(np.where(norm_roll < 0, plane_offset + 90, plane_offset - 90))
        elevation_angles = tuple(np.where(norm_roll < 0, norm_roll + 90, 90 - norm_roll))
        return azimuth_angles, elevation_angles


class CIPICPlane(InterauralPlaneDataset):
    def __init__(
        self,
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
        datapoint = CipicDataPoint(sofa_directory_path=Path(root)/'sofa', hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length, subject_ids, exclude_ids, target_spec, group_spec)


class ARIPlane(SphericalPlaneDataset):
    def __init__(
        self,
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
        datapoint = AriDataPoint(sofa_directory_path=Path(root)/'sofa', anthropomorphy_matfile_path=None, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, dtype=dtype)
        if positive_angles is None:
            positive_angles = False
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length, subject_ids, exclude_ids, target_spec, group_spec)


class ListenPlane(SphericalPlaneDataset):
    def __init__(
        self,
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
        datapoint = ListenDataPoint(sofa_directory_path=Path(root)/'sofa', hrtf_type=hrtf_type, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length, subject_ids, exclude_ids, target_spec, group_spec)


class BiLiPlane(SphericalPlaneDataset):
    def __init__(
        self,
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
        datapoint = BiLiDataPoint(sofa_directory_path=Path(root)/'sofa', hrtf_type=hrtf_type, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length, subject_ids, exclude_ids, target_spec, group_spec)


class ITAPlane(SphericalPlaneDataset):
    def __init__(
        self,
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
        datapoint = ItaDataPoint(sofa_directory_path=Path(root)/'sofa', hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length, subject_ids, exclude_ids, target_spec, group_spec)


class HUTUBSPlane(SphericalPlaneDataset):
    def __init__(
        self,
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
        datapoint = HutubsDataPoint(sofa_directory_path=Path(root)/'sofa', measured_hrtf=measured_hrtf, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length, subject_ids, exclude_ids, target_spec, group_spec)


class RIECPlane(SphericalPlaneDataset):
    def __init__(
        self,
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
        datapoint = RiecDataPoint(sofa_directory_path=Path(root)/'sofa', hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length, subject_ids, exclude_ids, target_spec, group_spec)


class CHEDARPlane(SphericalPlaneDataset):
    def __init__(
        self,
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
        datapoint = ChedarDataPoint(sofa_directory_path=Path(root)/'sofa', radius=radius, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length, subject_ids, exclude_ids, target_spec, group_spec)


class WidespreadPlane(SphericalPlaneDataset):
    def __init__(
        self,
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
        datapoint = WidespreadDataPoint(sofa_directory_path=Path(root)/'sofa', radius=radius, grid=grid, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length, subject_ids, exclude_ids, target_spec, group_spec)


class SADIE2Plane(SphericalPlaneDataset):
    def __init__(
        self,
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
        datapoint = Sadie2DataPoint(sofa_directory_path=Path(root)/'Database-Master_V1-4', hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length, subject_ids, exclude_ids, target_spec, group_spec)


class ThreeDThreeAPlane(SphericalPlaneDataset):
    def __init__(
        self,
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
        datapoint = ThreeDThreeADataPoint(sofa_directory_path=Path(root)/'HRTFs', hrtf_method=hrtf_method, hrtf_type=hrtf_type, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length, subject_ids, exclude_ids, target_spec, group_spec)


class SONICOMPlane(SphericalPlaneDataset):
    def __init__(
        self,
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
        datapoint = SonicomDataPoint(sofa_directory_path=Path(root), hrtf_type=hrtf_type, hrir_samplerate=hrir_samplerate, hrir_length=hrir_length, dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, hrir_samplerate, hrir_length, subject_ids, exclude_ids, target_spec, group_spec)
