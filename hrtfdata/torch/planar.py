from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Optional
import numpy as np
from .full import HRTFDataset
from ..core import DataPoint, AriDataPoint, ListenDataPoint, BiLiDataPoint, ItaDataPoint, HutubsDataPoint, RiecDataPoint, ChedarDataPoint, WidespreadDataPoint, Sadie2DataPoint, ThreeDThreeADataPoint
from ..util import wrap_closed_open_interval, wrap_open_closed_interval
from ..display import plot_hrir_plane, plot_hrtf_plane, plot_plane_angles


class PlaneTransform(ABC):

    def __init__(self, plane, plane_offset, positive_angles):
        self.plane = plane
        self.plane_offset = plane_offset
        self.positive_angles = positive_angles


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

        if self.closed_open_angles:
            plane_angles = wrap_closed_open_interval(plane_angles, self.min_angle, self.max_angle)
        else:
            plane_angles = wrap_open_closed_interval(plane_angles, self.min_angle, self.max_angle)
        return plane_angles


    def __repr__(self):
        return self.__class__.__name__ + '()'


    @staticmethod
    def _first_nonneg_angle_idx(angles):
        return np.flatnonzero(np.diff(np.sign(angles)))[0] + 1


    @staticmethod
    def _first_pos_angle_idx(angles):
        return np.flatnonzero(np.diff(np.sign(angles)))[-1] + 1


class SphericalPlaneTransform(PlaneTransform):
    def __init__(self, plane, plane_offset, positive_angles):
        super().__init__(plane, plane_offset, positive_angles)
        if positive_angles:
            self.min_angle = 0
            self.max_angle = 360
            self.closed_open_angles = True
        elif plane == 'horizontal':
            self.min_angle = -180
            self.max_angle = 180
            self.closed_open_angles = False
        else:
            self.min_angle = -90
            self.max_angle = 270
            self.closed_open_angles = True

    
    def calc_plane_angles(self, selected_angles):
        self.azimuth_angles = np.array(list(selected_angles.keys()))
        self.elevation_angles = list(selected_angles.values())[0]

        if self.plane == 'horizontal':
            input_angles = self.azimuth_angles
        elif len(self.azimuth_angles) > 1:
            other_elevation_angles = 180-list(selected_angles.values())[1]
            input_angles = [self.elevation_angles, other_elevation_angles]
            self._elevation_overlap90 = np.isclose(self.elevation_angles, 90).any() and np.isclose(other_elevation_angles, 90).any()
            self._elevation_overlap270 = np.isclose(self.elevation_angles, -90).any() and np.isclose(other_elevation_angles, 270).any()
        elif (self.plane == 'median' and np.isclose(self.azimuth_angles, [0]).any()
        or self.plane == 'frontal' and np.isclose(self.azimuth_angles, [90]).any()):
            # only first half plane is present
            input_angles = self.elevation_angles
        else:
            # only second half plane is present
            input_angles = 180-self.elevation_angles
        
        return super().calc_plane_angles(input_angles)


    def __call__(self, hrirs):
        if self.plane == 'horizontal':
            if self.positive_angles:
                split_idx = self._first_nonneg_angle_idx(self.azimuth_angles)
                left = hrirs[split_idx:]
                right = hrirs[:split_idx]
                single_plane = np.ma.concatenate((left, right))
            else:
                single_plane = hrirs
        elif self.plane == 'median':
            split_idx = self._first_nonneg_angle_idx(self.elevation_angles)
            if len(self.azimuth_angles) > 1:
                # both half planes present
                down_up_front = hrirs[0]
                up_down_back = np.flip(hrirs[1], axis=0)
                if self._elevation_overlap90:
                    down_up_front = down_up_front[:-1]
                if self._elevation_overlap270:
                    up_down_back = up_down_back[:-1]
                if self.positive_angles:
                    down_front, up_front = np.split(down_up_front, [split_idx])
                    single_plane = np.ma.concatenate((up_front, up_down_back, down_front))
                else:
                    single_plane = np.ma.concatenate((down_up_front, up_down_back))
            elif np.isclose(self.azimuth_angles, 0).any():
                # only front half plane present
                down_up_front = hrirs
                if self.positive_angles:
                    down_front, up_front = np.split(down_up_front, [split_idx])
                    single_plane = np.ma.concatenate((up_front, down_front))
                else:
                    single_plane = down_up_front
            else:
                # only back half plane present
                up_down_back = np.flip(hrirs, axis=0)
                single_plane = up_down_back
        elif self.plane == 'frontal':
            if len(self.azimuth_angles) > 1:
                # both half planes present
                left_down_up = hrirs[0]
                right_up_down = np.flip(hrirs[1], axis=0)
                if self._elevation_overlap90:
                    left_down_up = left_down_up[:-1]
                if self._elevation_overlap270:
                    right_up_down = right_up_down[:-1]
                if self.positive_angles:
                    split_idx = self._first_nonneg_angle_idx(self.elevation_angles)
                    left_down, left_up = np.split(left_down_up, [split_idx])
                    single_plane = np.ma.concatenate((left_up, right_up_down, left_down))
                else:
                    single_plane = np.ma.concatenate((left_down_up, right_up_down))
            elif np.isclose(self.azimuth_angles, 90).any():
                # only left half plane present
                left_down_up = hrirs
                if self.positive_angles:
                    split_idx = self._first_nonneg_angle_idx(self.elevation_angles)
                    left_down, left_up = np.split(left_down_up, [split_idx])
                    single_plane = np.ma.concatenate((left_up, left_down))
                else:
                    single_plane = left_down_up
            else:
                # only right half plane present
                right_up_down = np.flip(hrirs, axis=0)
                single_plane = right_up_down

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
        subject_ids=None,
        subject_requirements = None,
    ):
        self._plane = plane
        self._plane_offset = plane_offset
        if plane not in ('horizontal', 'median', 'frontal'):
            raise ValueError('Unknown plane "{}", needs to be "horizontal", "median" or "frontal".')
        self._domain = domain

        feature_spec = {'hrirs': {'row_angles': row_angles, 'column_angles': column_angles, 'side': side, 'domain': domain}}
        label_spec = {'side': {}}
        super().__init__(datapoint, feature_spec, label_spec, subject_ids, subject_requirements, hrir_transform=planar_transform)
        self.plane_angles = planar_transform.calc_plane_angles(self._selected_angles)
        self.min_angle = planar_transform.min_angle
        self.max_angle = planar_transform.max_angle
        self.closed_open_angles = planar_transform.closed_open_angles


    def plot_plane(self, idx, ax=None, cmap='viridis', continuous=False, vmin=None, vmax=None, title=None, colorbar=True, log_freq=False):
        if self._plane == 'horizontal':
            angles_label = 'yaw [°]'
        elif self._plane == 'median':
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
                ' With Offset {}°'.format(self._plane_offset) if self._plane_offset != 0 else '',
                item['group'],
                item['target'].replace('-', ' ').title(),
            )
        ax.set_title(title)
        return ax


    def plot_angles(self, ax=None, title=None):
        if self._plane == 'horizontal':
            zero_location = 'N'
        else:
            zero_location = 'E'
        ax = plot_plane_angles(self.plane_angles, self.min_angle, self.max_angle, self.closed_open_angles, 1, zero_location, ax) # TODO use actual radius
        if title is None:
            title = 'Angles in the {} Plane'.format(self._plane.title())
        ax.set_title(title)
        return ax


class SphericalPlaneDataset(HRTFPlaneDataset):
    def __init__(self,
        datapoint: DataPoint,
        plane: str,
        domain: str = 'magnitude_db',
        side: str = 'both-left',
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
    ):
        if plane == 'horizontal':
            azimuth_angles = plane_angles
            elevation_angles = [plane_offset]
        elif plane == 'median':
            if plane_offset != 0:
                raise ValueError('Only the median plane at azimuth 0 is available in a spherical coordinate dataset')
            azimuth_angles = [0, 180]
            elevation_angles = plane_angles
        elif plane == 'frontal':
            if plane_offset != 0:
                raise ValueError('Only the frontal plane at azimuth +/-90 is available in a spherical coordinate dataset')
            azimuth_angles = [-90, 90]
            elevation_angles = plane_angles

        plane_transform = SphericalPlaneTransform(plane, plane_offset, positive_angles)
        super().__init__(datapoint, plane, domain, side, plane_offset, azimuth_angles, elevation_angles, plane_transform)


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
    ):
        datapoint = AriDataPoint(sofa_directory_path=Path(root)/'sofa', anthropomorphy_matfile_path=None)
        if positive_angles is None:
            positive_angles = False
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles)


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
    ):
        datapoint = ListenDataPoint(sofa_directory_path=Path(root)/'sofa/compensated/44100')
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles)


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
    ):
        datapoint = BiLiDataPoint(sofa_directory_path=Path(root)/'sofa/compensated/96000')
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles)


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
    ):
        datapoint = ItaDataPoint(sofa_directory_path=Path(root)/'sofa')
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles)


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
    ):
        datapoint = HutubsDataPoint(sofa_directory_path=Path(root)/'sofa')
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles)


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
    ):
        datapoint = RiecDataPoint(sofa_directory_path=Path(root)/'sofa')
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles)


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
    ):
        datapoint = ChedarDataPoint(sofa_directory_path=Path(root)/'sofa')
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles)


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
    ):
        datapoint = WidespreadDataPoint(sofa_directory_path=Path(root)/'sofa')
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles)


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
    ):
        datapoint = Sadie2DataPoint(sofa_directory_path=Path(root)/'Database-Master_V1-4')
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles)


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
    ):
        datapoint = ThreeDThreeADataPoint(sofa_directory_path=Path(root)/'sofa')
        super().__init__(datapoint, plane, domain, side, plane_angles, plane_offset, positive_angles)
