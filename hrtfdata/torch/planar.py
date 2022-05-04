from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Optional
import numpy as np
from .full import HRTFDataset
from ..datapoint import DataPoint, CipicDataPoint, AriDataPoint, ListenDataPoint, BiLiDataPoint, ItaDataPoint, HutubsDataPoint, RiecDataPoint, ChedarDataPoint, WidespreadDataPoint, Sadie2DataPoint, ThreeDThreeADataPoint, SonicomDataPoint
from ..util import wrap_closed_open_interval, wrap_open_closed_interval
from ..display import plot_hrir_plane, plot_hrtf_plane, plot_plane_angles


class PlaneTransform(ABC):

    def __init__(self, plane, plane_offset, positive_angles):
        self.plane = plane
        self.plane_offset = plane_offset
        self.positive_angles = positive_angles


    @property
    def positive_angles(self):
        return self._positive_angles


    @positive_angles.setter
    def positive_angles(self, value):
        self._positive_angles = value
        if value:
            self.min_angle = 0
            self.max_angle = 360
        elif self.plane in ('horizontal', 'interaural', 'frontal'):
            self.min_angle = -180
            self.max_angle = 180
        else:
            self.min_angle = -90
            self.max_angle = 270


    @abstractmethod
    def __call__(self, single_plane):
        try:
            if single_plane.mask.any():
                if single_plane.ndim > 1:
                    keep_angles = ~single_plane.mask.all(axis=-1)
                else:
                    keep_angles = ~single_plane.mask
                single_plane = single_plane[keep_angles]
            return np.ma.getdata(single_plane)
        except AttributeError:
            return single_plane


    @abstractmethod
    def calc_plane_angles(self, input_angles):
        plane_angles = self(input_angles)
        return wrap_closed_open_interval(plane_angles, self.min_angle, self.max_angle)


    def __repr__(self):
        return self.__class__.__name__ + '()'


    @staticmethod
    def _idx_first_nonneg(iterable):
        return np.flatnonzero(np.diff(np.sign(iterable)))[0] + 1


    @staticmethod
    def _idx_first_pos(iterable):
        return np.flatnonzero(np.diff(np.sign(iterable)))[-1] + 1


    @staticmethod
    def _idx_first_not_smaller_than(iterable, value=0):
        return np.flatnonzero(iterable >= value)[0]


    @staticmethod
    def _idx_first_larger_than(iterable, value=0):
        return np.flatnonzero(iterable > value)[0]


class InterauralPlaneTransform(PlaneTransform):

    _split_idx: int
    _left_pole_overlap: bool
    _right_pole_overlap: bool
    _pos_sphere_present: bool
    _neg_sphere_present: bool


    def calc_plane_angles(self, selected_angles):
        if not selected_angles:
            return np.array([])
        vertical_angles = np.array(list(selected_angles.keys()))
        lateral_angles = list(selected_angles.values())[0].copy()

        if self.plane == 'median':
            input_angles = vertical_angles
            if self.positive_angles:
                self._split_idx = self._idx_first_nonneg(vertical_angles)
            else:
                self._split_idx = self._idx_first_not_smaller_than(vertical_angles, -90)
        else:
            self._split_idx = self._idx_first_nonneg(np.ma.getdata(lateral_angles))
            if len(vertical_angles) > 1:
                # both half planes present
                back_yaw_angles = 180-lateral_angles
                front_yaw_angles = list(selected_angles.values())[1].copy()
                input_angles = [back_yaw_angles, front_yaw_angles]
                self._left_pole_overlap = np.isclose(front_yaw_angles, 90).any() and np.isclose(back_yaw_angles, 90).any()
                self._right_pole_overlap = np.isclose(front_yaw_angles, -90).any() and np.isclose(back_yaw_angles, 270).any()
                self._pos_sphere_present = True
                self._neg_sphere_present = True
                if self.plane == 'frontal':
                    # reverse angular direction
                    input_angles = [-x for x in input_angles]
            else:
                # floating point safe version of check below
                # if vertical_angles <= 90 and vertical_angles > -90:
                rtol = 1e-5
                atol = 1e-8
                upper_lim = vertical_angles - 90
                lower_lim = vertical_angles + 90
                if (upper_lim <= rtol*np.abs(upper_lim)+atol and lower_lim > rtol*np.abs(lower_lim)+atol).item():
                    # only front half plane is present
                    self._pos_sphere_present = True
                    self._neg_sphere_present = False
                    input_angles = lateral_angles
                else:
                    # only back half plane is present
                    self._pos_sphere_present = False
                    self._neg_sphere_present = True
                    input_angles = 180-lateral_angles
                self._left_pole_overlap = False
                self._right_pole_overlap = False
                if self.plane == 'frontal':
                    input_angles = -input_angles # reverse angular direction

        return super().calc_plane_angles(input_angles)


    def __call__(self, hrirs):
        if self.plane == 'median':
            if self.positive_angles:
                down, up = np.split(hrirs, [self._split_idx])
                single_plane = np.ma.concatenate((up, down))
            else:
                back_down, rest = np.split(hrirs, [self._split_idx])
                single_plane = np.ma.concatenate((rest, back_down))
        else:
            if self._left_pole_overlap:
                left_pole_mask = np.full_like(hrirs[1], False, dtype=bool)
                left_pole_mask[-1] = True
                hrirs[1] = np.ma.masked_where(left_pole_mask, hrirs[1], False)
            if self._right_pole_overlap:
                right_pole_mask = np.full_like(hrirs[0], False, dtype=bool)
                right_pole_mask[0] = True
                hrirs[0] = np.ma.masked_where(right_pole_mask, hrirs[0], False)
            if self.plane == 'frontal':
                if self._pos_sphere_present and self._neg_sphere_present:
                    # both half planes present
                    down_right_left = hrirs[0]
                    up_left_right = np.flip(hrirs[1], axis=0)
                    if self.positive_angles:
                        left_up, right_up = np.split(up_left_right, [-self._split_idx-1])
                        single_plane = np.ma.concatenate((right_up, down_right_left, left_up))
                    else:
                        right_down, left_down = np.split(down_right_left, [self._split_idx])
                        single_plane = np.ma.concatenate((left_down, up_left_right, right_down))
                elif self._pos_sphere_present:
                    # only up half plane present
                    up_left_right = np.flip(hrirs, axis=0)
                    if self.positive_angles:
                        left_up, right_up = np.split(up_left_right, [-self._split_idx-1])
                        single_plane = np.ma.concatenate((right_up, left_up))
                    else:
                        single_plane = up_left_right
                else:
                    # only down half plane present
                    down_right_left = hrirs
                    if self.positive_angles:
                        single_plane = down_right_left
                    else:
                        right_down, left_down = np.split(down_right_left, [self._split_idx])
                        single_plane = np.ma.concatenate((left_down, right_down))
            else:
                if self._pos_sphere_present and self._neg_sphere_present:
                    # both half planes present
                    back_left_right = np.flip(hrirs[0], axis=0)
                    front_right_left = hrirs[1]
                    if self.positive_angles:
                        front_right, front_left = np.split(front_right_left, [self._split_idx])
                        single_plane = np.ma.concatenate((front_left, back_left_right, front_right))
                    else:
                        back_left, back_right = np.split(back_left_right, [-self._split_idx-1])
                        single_plane = np.ma.concatenate((back_right, front_right_left, back_left))
                elif self._pos_sphere_present:
                    # only front half plane present
                    front_right_left = hrirs
                    if self.positive_angles:
                        front_right, front_left = np.split(front_right_left, [self._split_idx])
                        single_plane = np.ma.concatenate((front_left, front_right))
                    else:
                        single_plane = front_right_left
                else:
                    # only back half plane present
                    back_left_right = np.flip(hrirs, axis=0)
                    if self.positive_angles:
                        single_plane = back_left_right
                    else:
                        back_left, back_right = np.split(back_left_right, [-self._split_idx-1])
                        single_plane = np.ma.concatenate((back_right, back_left))


        return super().__call__(single_plane)


class SphericalPlaneTransform(PlaneTransform):

    _split_idx: int
    _up_pole_overlap: bool
    _down_pole_overlap: bool
    _pos_sphere_present: bool
    _neg_sphere_present: bool

    
    def calc_plane_angles(self, selected_angles):
        if not selected_angles:
            return np.array([])
        azimuth_angles = np.array(list(selected_angles.keys()))
        elevation_angles = list(selected_angles.values())[0].copy()

        if self.plane == 'horizontal':
            input_angles = azimuth_angles
            self._split_idx = self._idx_first_nonneg(azimuth_angles)
        else:
            self._split_idx = self._idx_first_nonneg(np.ma.getdata(elevation_angles))
            if len(azimuth_angles) > 1:
                # both half planes present
                back_pitch_angles = 180-elevation_angles
                front_pitch_angles = list(selected_angles.values())[1].copy()
                input_angles = [back_pitch_angles, front_pitch_angles]
                self._up_pole_overlap = np.isclose(front_pitch_angles, 90).any() and np.isclose(back_pitch_angles, 90).any()
                self._down_pole_overlap = np.isclose(front_pitch_angles, -90).any() and np.isclose(back_pitch_angles, 270).any()
                self._pos_sphere_present = True
                self._neg_sphere_present = True
                if self.plane == 'frontal':
                    # shift origin from Y to Z axis
                    input_angles = [x - 90 for x in input_angles]
            else:
                # floating point safe version of check below
                # if azimuth_angles <= 90 and azimuth_angles > -90:
                rtol = 1e-5
                atol = 1e-8
                upper_lim = azimuth_angles - 90
                lower_lim = azimuth_angles + 90
                if (upper_lim <= rtol*np.abs(upper_lim)+atol and lower_lim > rtol*np.abs(lower_lim)+atol).item():
                    # only front half plane is present
                    self._pos_sphere_present = True
                    self._neg_sphere_present = False
                    input_angles = elevation_angles
                else:
                    # only back half plane is present
                    self._pos_sphere_present = False
                    self._neg_sphere_present = True
                    input_angles = 180-elevation_angles
                self._up_pole_overlap = False
                self._down_pole_overlap = False
                if self.plane == 'frontal':
                    input_angles -= 90 # shift origin from Y to Z axis
        
        return super().calc_plane_angles(input_angles)


    def __call__(self, hrirs):
        if self.plane == 'horizontal':
            if self.positive_angles:
                right, left = np.split(hrirs, [self._split_idx])
                single_plane = np.ma.concatenate((left, right))
            else:
                single_plane = hrirs
        else:
            if self._up_pole_overlap:
                up_pole_mask = np.full_like(hrirs[1], False, dtype=bool)
                up_pole_mask[-1] = True
                hrirs[1] = np.ma.masked_where(up_pole_mask, hrirs[1], False)
            if self._down_pole_overlap:
                down_pole_mask = np.full_like(hrirs[0], False, dtype=bool)
                down_pole_mask[0] = True
                hrirs[0] = np.ma.masked_where(down_pole_mask, hrirs[0], False)
            if self.plane == 'frontal':
                if self._pos_sphere_present and self._neg_sphere_present:
                    # both right and left half planes present
                    right_up_down = np.flip(hrirs[0], axis=0)
                    left_down_up = hrirs[1]
                    if self.positive_angles:
                        single_plane = np.ma.concatenate((right_up_down, left_down_up))
                    else:
                        single_plane = np.ma.concatenate((left_down_up, right_up_down))
                elif self._pos_sphere_present:
                    # only left half plane present
                    left_down_up = hrirs
                    single_plane = left_down_up
                else:
                    # only right half plane present
                    right_up_down = np.flip(hrirs, axis=0)
                    single_plane = right_up_down
            else:
                if self._pos_sphere_present and self._neg_sphere_present:
                    # both back and front half planes present
                    back_up_down = np.flip(hrirs[0], axis=0)
                    front_down_up = hrirs[1]
                    if not self.positive_angles:
                        single_plane = np.ma.concatenate((front_down_up, back_up_down))
                    else:
                        front_down, front_up = np.split(front_down_up, [self._split_idx])
                        single_plane = np.ma.concatenate((front_up, back_up_down, front_down))
                elif self._pos_sphere_present:
                    # only front half plane present
                    front_down_up = hrirs
                    if self.positive_angles:
                        front_down, front_up = np.split(front_down_up, [self._split_idx])
                        single_plane = np.ma.concatenate((front_up, front_down))
                    else:
                        single_plane = front_down_up
                else:
                    # only back half plane present
                    back_up_down = np.flip(hrirs, axis=0)
                    single_plane = back_up_down

        return super().__call__(single_plane)


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
        subject_ids: Optional[Iterable[int]] = None,
    ):
        self._plane = plane
        self._plane_offset = plane_offset
        if plane not in ('horizontal', 'median', 'frontal', 'vertical', 'interaural'):
            raise ValueError('Unknown plane "{}", needs to be "horizontal", "median", "frontal", "vertical" or "interaural".')
        self._domain = domain
        self._planar_transform = planar_transform

        feature_spec = {'hrirs': {'row_angles': row_angles, 'column_angles': column_angles, 'side': side, 'domain': domain}}
        target_spec = {'side': {}}
        group_spec = {'subject': {}}
        super().__init__(datapoint, feature_spec, target_spec, group_spec, subject_ids, hrir_transform=planar_transform)
        self.positive_angles = planar_transform.positive_angles


    @property
    def positive_angles(self):
        return self._planar_transform.positive_angles


    @positive_angles.setter
    def positive_angles(self, value):
        self._planar_transform.positive_angles = value
        self.plane_angles = self._planar_transform.calc_plane_angles(self._selected_angles)
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
                item['group'],
                item['target'].replace('-', ' ').title(),
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
        subject_ids: Optional[Iterable[int]] = None,
    ):
        if plane == 'horizontal':
            if plane_offset != 0:
                raise ValueError('Only the horizontal plane at vertical angle 0 is available in an interaural coordinate dataset')
            lateral_angles = plane_angles
            vertical_angles = [-180, 0]
        elif plane == 'median':
            lateral_angles = [plane_offset]
            vertical_angles = plane_angles
        elif plane == 'frontal':
            if plane_offset != 0:
                raise ValueError('Only the frontal plane at vertical angles +/-90 is available in an interaural coordinate dataset')
            lateral_angles = plane_angles
            vertical_angles = [-90, 90]
        elif plane == 'interaural':
            lateral_angles = plane_angles
            vertical_angles = wrap_closed_open_interval([plane_offset-180, plane_offset], -180, 180)
        else:
            if plane not in ('horizontal', 'median', 'frontal', 'interaural'):
                raise ValueError('Unknown plane "{}", needs to be "horizontal", "median", "frontal" or "interaural".')

        plane_transform = InterauralPlaneTransform(plane, plane_offset, positive_angles)
        super().__init__(datapoint, plane, domain, side, plane_offset, vertical_angles, lateral_angles, plane_transform, subject_ids)


class SphericalPlaneDataset(HRTFPlaneDataset):
    def __init__(self,
        datapoint: DataPoint,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'both-left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        subject_ids: Optional[Iterable[int]] = None,
    ):
        if plane == 'horizontal':
            azimuth_angles = plane_angles
            elevation_angles = [plane_offset]
        elif plane == 'median':
            if plane_offset != 0:
                raise ValueError('Only the median plane at azimuth 0 is available in a spherical coordinate dataset')
            azimuth_angles = [-180, 0]
            elevation_angles = plane_angles
        elif plane == 'frontal':
            if plane_offset != 0:
                raise ValueError('Only the frontal plane at azimuth +/-90 is available in a spherical coordinate dataset')
            azimuth_angles = [-90, 90]
            elevation_angles = plane_angles
        elif plane == 'vertical':
            azimuth_angles = wrap_closed_open_interval([plane_offset-180, plane_offset], -180, 180)
            elevation_angles = plane_angles
        else:
            if plane not in ('horizontal', 'median', 'frontal', 'vertical'):
                raise ValueError('Unknown plane "{}", needs to be "horizontal", "median", "frontal" or "vertical".')

        plane_transform = SphericalPlaneTransform(plane, plane_offset, positive_angles)
        super().__init__(datapoint, plane, domain, side, plane_offset, azimuth_angles, elevation_angles, plane_transform, subject_ids)



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
        subject_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
    ):
        datapoint = CipicDataPoint(sofa_directory_path=Path(root)/'sofa', dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, subject_ids)


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
        subject_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
    ):
        datapoint = AriDataPoint(sofa_directory_path=Path(root)/'sofa', anthropomorphy_matfile_path=None, dtype=dtype)
        if positive_angles is None:
            positive_angles = False
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, subject_ids)


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
        subject_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
    ):
        datapoint = ListenDataPoint(sofa_directory_path=Path(root)/'sofa/compensated/44100', dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, subject_ids)


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
        subject_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
    ):
        datapoint = BiLiDataPoint(sofa_directory_path=Path(root)/'sofa/compensated/96000', dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, subject_ids)


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
        subject_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
    ):
        datapoint = ItaDataPoint(sofa_directory_path=Path(root)/'sofa', dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, subject_ids)


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
        subject_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
    ):
        datapoint = HutubsDataPoint(sofa_directory_path=Path(root)/'sofa', dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, subject_ids)


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
        subject_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
    ):
        datapoint = RiecDataPoint(sofa_directory_path=Path(root)/'sofa', dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, subject_ids)


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
        subject_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
    ):
        datapoint = ChedarDataPoint(sofa_directory_path=Path(root)/'sofa', dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, subject_ids)


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
        subject_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
    ):
        datapoint = WidespreadDataPoint(sofa_directory_path=Path(root)/'sofa', dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, subject_ids)


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
        subject_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
    ):
        datapoint = Sadie2DataPoint(sofa_directory_path=Path(root)/'Database-Master_V1-4', dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, subject_ids)


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
        subject_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
    ):
        datapoint = ThreeDThreeADataPoint(sofa_directory_path=Path(root)/'sofa', dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, subject_ids)


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
        subject_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
    ):
        datapoint = SonicomDataPoint(sofa_directory_path=Path(root), dtype=dtype)
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles, subject_ids)
