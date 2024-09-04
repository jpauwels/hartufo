from .query import DataQuery, CipicDataQuery, AriDataQuery, ListenDataQuery, BiLiDataQuery, CrossModDataQuery, ItaDataQuery, HutubsDataQuery, RiecDataQuery, ChedarDataQuery, WidespreadDataQuery, Sadie2DataQuery, Princeton3D3ADataQuery, ScutDataQuery, SonicomDataQuery, MitKemarDataQuery, CustomDataQuery
from .transforms.hrir import InterauralPlaneTransform, SphericalPlaneTransform
from .util import wrap_closed_open_interval, wrap_closed_interval, spherical2cartesian, spherical2interaural, cartesian2spherical, cartesian2interaural, interaural2spherical, interaural2cartesian, quantise
from abc import abstractmethod
from pathlib import Path
from typing import Iterable, Optional, Union
import numpy as np
import netCDF4 as ncdf
from PIL import Image


class DataReader:

    def __init__(self,
        query: DataQuery,
    ):
        self.query = query


class SofaDataReader(DataReader):
    """
    An abstract class that reads the HRIR for a given subject id of a dataset from a SOFA file and stores it internally
    as a 3D tensor. The HRIR directions are stored in the rows and columns as a plate carrée projection where each
    column represents a plane parallel to the fundamental plane, i.e. each row represents a single angle in the
    fundamental plane. The poles (if present) are therefore stored in the first and last column.

    fundamental_angle: angle in the fundamental plane with range  [-180, 180)
    (azimuth for spherical coordinates, vertical angle for interaural coordinates)
    orthogonal_angle: angle between fundamental plane and directional vector with range [-90, 90]
    (elevation for spherical coordinates, lateral angle for interaural coordinates)
    """

    _angle_quantisation: int = -3
    _distance_quantisation: int = -3


    @property
    @abstractmethod
    def fundamental_angle_name(self):
        pass


    @property
    @abstractmethod
    def orthogonal_angle_name(self):
        pass


    @abstractmethod
    def _sofa_path(self, subject_id):
        pass


    @staticmethod
    @abstractmethod
    def _convert_positions(coordinate_system, positions):
        pass


    @staticmethod
    @abstractmethod
    def _coordinate_transform(coordinate_system, selected_fundamental_angles, selected_orthogonal_angles, selected_radii):
        pass


    @staticmethod
    @abstractmethod
    def _mirror_hrirs(hrirs, selected_fundamental_angles):
        pass


    @staticmethod
    @abstractmethod
    def _verify_angle_symmetry(fundamental_angles, orthogonal_angles):
        pass


    def hrir_samplerate(self, subject_id):
        sofa_path = self._sofa_path(subject_id)
        hrir_file = ncdf.Dataset(sofa_path)
        try:
            samplerate = hrir_file.variables['Data.SamplingRate'][:].item()
        except Exception as exc:
            raise ValueError(f'Error reading file "{sofa_path}"') from exc
        finally:
            hrir_file.close()
        return samplerate


    def hrir_length(self, subject_id):
        sofa_path = self._sofa_path(subject_id)
        hrir_file = ncdf.Dataset(sofa_path)
        try:
            length = hrir_file.dimensions['N'].size
        except Exception as exc:
            raise ValueError(f'Error reading file "{sofa_path}"') from exc
        finally:
            hrir_file.close()
        return length


    def _map_sofa_position_order_to_matrix(self, subject_id, fundamental_angles, orthogonal_angles, distance):
        if fundamental_angles is not None and orthogonal_angles is not None and len(fundamental_angles) != len(orthogonal_angles):
            raise ValueError(f'The number of fundamental angles ({len(fundamental_angles)}) differs from the number of orthogonal angles ({len(orthogonal_angles)})')
        sofa_path = self._sofa_path(subject_id)
        hrir_file = ncdf.Dataset(sofa_path)
        try:
            positions = self._convert_positions(
                hrir_file.variables['SourcePosition'].Type,
                np.ma.getdata(hrir_file.variables['SourcePosition'][:]),
            )
        except Exception as exc:
            raise ValueError(f'Error reading file "{sofa_path}"') from exc
        finally:
            hrir_file.close()
        quantised_positions =  np.column_stack((quantise(positions[:, :2], self._angle_quantisation), 
                                                quantise(positions[:, 2], self._distance_quantisation)))
        quantised_positions[:, 0] = wrap_closed_open_interval(quantised_positions[:, 0], -180, 180)
        if isinstance(distance, str):
            file_radii = np.unique(quantised_positions[:, 2])
            if distance == 'nearest':
                distance = file_radii[0]
            elif distance == 'farthest':
                distance = file_radii[-1]
            else:
                raise ValueError(f'Invalid distance "{distance}". Only "nearest", "farthest" or a value in meter is allowed.')

        if fundamental_angles is not None:
            fundamental_angles = wrap_closed_open_interval(fundamental_angles, -180, 180)
        if orthogonal_angles is not None:
            orthogonal_angles = wrap_closed_interval(orthogonal_angles, -90, 90)
        if fundamental_angles is None and orthogonal_angles is None and distance is None:
            # Read all positions in file, without extra checks
            selected_file_indices = list(range(len(quantised_positions)))
        else:
            # Check if file positions are part of requested positions
            selected_file_indices = []
            if fundamental_angles is None:
                check_fundamental_angle = lambda _: True
            else:
                check_fundamental_angle = lambda file_fundamental_angle: np.isclose(file_fundamental_angle, fundamental_angles)
            if orthogonal_angles is None:
                check_orthogonal_angle = lambda _: True
            else:
                check_orthogonal_angle = lambda file_orthogonal_angle: np.isclose(file_orthogonal_angle, orthogonal_angles)
            if distance is None:
                check_distance = lambda _: True
            else:
                check_distance = lambda file_radius: np.isclose(file_radius, distance)
            for file_idx, (file_fundamental_angle, file_orthogonal_angle, file_radius) in enumerate(quantised_positions):
                if (check_fundamental_angle(file_fundamental_angle) & check_orthogonal_angle(file_orthogonal_angle) & check_distance(file_radius)).any():
                    selected_file_indices.append(file_idx)

        selected_positions = quantised_positions[selected_file_indices]
        selected_fundamental_angles = selected_positions[:, 0]
        selected_orthogonal_angles = selected_positions[:, 1]
        selected_radii = selected_positions[:, 2]
        unique_radii = np.unique(selected_radii)

        for pole_angle in (-90, 90):
            # If value at pole requested
            if orthogonal_angles is None or np.isclose(orthogonal_angles, pole_angle).any():
                for radius in unique_radii:
                    pole_indices = np.flatnonzero(np.isclose(quantised_positions[:, 1], pole_angle) & (quantised_positions[:, 2] == radius))
                    # If at least one value at pole present in file
                    if len(pole_indices) > 0:
                        # Make sure to include all requested row angles at pole
                        if fundamental_angles is not None and orthogonal_angles is not None:
                            requested_fundamental_angles_at_pole = fundamental_angles[orthogonal_angles == pole_angle]
                            selected_fundamental_angles = np.concatenate((selected_fundamental_angles, requested_fundamental_angles_at_pole))
                        # If pole angle not present in selection yet
                        if len(selected_orthogonal_angles) == 0 or not np.isclose(selected_orthogonal_angles, pole_angle).any():
                            # Add to column angles at appropriate extremum
                            if pole_angle > 0 or len(selected_orthogonal_angles) == 0:
                                selected_orthogonal_angles = np.append(selected_orthogonal_angles, pole_angle)
                            else:
                                selected_orthogonal_angles = np.insert(selected_orthogonal_angles, 0, pole_angle)

        unique_fundamental_angles = np.unique(selected_fundamental_angles)
        unique_orthogonal_angles = np.unique(selected_orthogonal_angles)

        additional_pole_indices = []

        def repeat_value_at_pole(pole_angle, pole_column_idx):
            # If value at pole requested
            if orthogonal_angles is None or np.isclose(orthogonal_angles, pole_angle).any():
                for radius_idx, radius in enumerate(unique_radii):
                    pole_indices = np.flatnonzero(np.isclose(quantised_positions[:, 1], pole_angle) & (quantised_positions[:, 2] == radius))
                    # If at least one value at pole present in file
                    if len(pole_indices) > 0:
                        pole_idx = pole_indices[0]
                        # Copy to those row angles that miss the value at the pole (if any)
                        radius_positions = selected_positions[selected_positions[:, 2] == radius]
                        present_fundamental_angles = radius_positions[np.isclose(radius_positions[:, 1], pole_angle), 0]
                        missing_fundamental_angles = np.setdiff1d(unique_fundamental_angles, present_fundamental_angles)
                        missing_row_indices = [np.argmax(angle == unique_fundamental_angles) for angle in missing_fundamental_angles]
                        selected_file_indices.extend([pole_idx] * len(missing_row_indices))
                        additional_pole_indices.extend([(row_idx, pole_column_idx, radius_idx) for row_idx in missing_row_indices])

        repeat_value_at_pole(-90, 0)
        repeat_value_at_pole(90, -1)

        selection_mask_indices = []
        for file_fundamental_angle, file_orthogonal_angle, file_radius in selected_positions:
            selection_mask_indices.append((np.argmax(file_fundamental_angle == unique_fundamental_angles), np.argmax(file_orthogonal_angle == unique_orthogonal_angles), np.argmax(file_radius == unique_radii)))
        selection_mask_indices.extend(additional_pole_indices)
        if len(selection_mask_indices) == 0:
            raise ValueError('None of the specified HRIR measurement positions are available in this dataset')

        selection_mask_indices = tuple(np.array(selection_mask_indices).T)
        selection_mask = np.full((len(unique_fundamental_angles), len(unique_orthogonal_angles), len(unique_radii)), True)
        selection_mask[selection_mask_indices] = False

        return unique_fundamental_angles, unique_orthogonal_angles, unique_radii, selection_mask, selected_file_indices, selection_mask_indices


    def hrir_positions(self, subject_id, coordinate_system, fundamental_angles=None, orthogonal_angles=None, distance=None):
        selected_fundamental_angles, selected_orthogonal_angles, selected_radii, selection_mask, *_ = self._map_sofa_position_order_to_matrix(subject_id, fundamental_angles, orthogonal_angles, distance)

        coordinates = self._coordinate_transform(coordinate_system, selected_fundamental_angles, selected_orthogonal_angles, selected_radii)

        position_grid = np.stack(np.meshgrid(*coordinates, indexing='ij'), axis=-1)
        if selection_mask.any(): # sparse grid
            tiled_position_mask = np.tile(selection_mask[:, :, :, np.newaxis], (1, 1, 1, 3))
            return np.ma.masked_where(tiled_position_mask, position_grid)
        # dense grid
        return position_grid


    def hrir(self, subject_id, side, fundamental_angles=None, orthogonal_angles=None, distance=None):
        sofa_path = self._sofa_path(subject_id)
        hrir_file = ncdf.Dataset(sofa_path)
        try:
            hrirs = np.ma.getdata(hrir_file.variables['Data.IR'][:, 0 if side.endswith('left') else 1, :])
            samplerate = hrir_file.variables['Data.SamplingRate'][:].item()
        except Exception as exc:
            raise ValueError(f'Error reading file "{sofa_path}"') from exc
        finally:
            hrir_file.close()
        selected_fundamental_angles, _, _, selection_mask, selected_file_indices, selection_mask_indices = self._map_sofa_position_order_to_matrix(subject_id, fundamental_angles, orthogonal_angles, distance)
        hrir_matrix = np.empty(selection_mask.shape + (hrirs.shape[1],))
        hrir_matrix[selection_mask_indices] = hrirs[selected_file_indices]
        tiled_selection_mask = np.tile(selection_mask[:, :, :, np.newaxis], (1, 1, 1, hrir_matrix.shape[-1]))
        hrir = np.ma.masked_where(tiled_selection_mask, hrir_matrix, copy=False)
        if side.startswith('mirrored'):
            return self._mirror_hrirs(hrir, selected_fundamental_angles)
        return hrir


class SofaSphericalDataReader(SofaDataReader):
    PlaneTransform = SphericalPlaneTransform


    @property
    def fundamental_angle_name(self):
        return 'azimuth [°]'


    @property
    def orthogonal_angle_name(self):
        return 'elevation [°]'


    def hrir_positions(self, subject_id, fundamental_angles=None, orthogonal_angles=None, coordinate_system='spherical'):
        return super().hrir_positions(subject_id, coordinate_system, fundamental_angles, orthogonal_angles)


    @staticmethod
    def _convert_positions(coordinate_system, positions):
        if coordinate_system == 'cartesian':
            positions = np.stack(cartesian2spherical(*positions.T), axis=1)
        return positions


    @staticmethod
    def _coordinate_transform(coordinate_system, selected_fundamental_angles, selected_orthogonal_angles, selected_radii):
        if coordinate_system == 'spherical':
            return selected_fundamental_angles, selected_orthogonal_angles, selected_radii
        if coordinate_system == 'interaural':
            return spherical2interaural(selected_fundamental_angles, selected_orthogonal_angles, selected_radii)
        if coordinate_system == 'cartesian':
            return spherical2cartesian(selected_fundamental_angles, selected_orthogonal_angles, selected_radii)
        raise ValueError(f'Unknown coordinate system "{coordinate_system}"')


    @staticmethod
    def _mirror_hrirs(hrirs, selected_fundamental_angles):
        # flip azimuths (in rows)
        if np.isclose(selected_fundamental_angles[0], -180):
            return np.ma.row_stack((hrirs[0:1], np.flipud(hrirs[1:])))
        else:
            return np.flipud(hrirs)


    @staticmethod
    def _verify_angle_symmetry(fundamental_angles, _):
        # mirror azimuths/rows
        start_idx = 1 if np.isclose(fundamental_angles[0], -180) else 0
        if not np.allclose(fundamental_angles[start_idx:], -np.flip(fundamental_angles[start_idx:])):
            raise ValueError('Only datasets with symmetric azimuths can mix mirrored and non-mirrored sides.')


class SofaInterauralDataReader(SofaDataReader):
    PlaneTransform = InterauralPlaneTransform


    @property
    def fundamental_angle_name(self):
        return 'vertical [°]'


    @property
    def orthogonal_angle_name(self):
        return 'lateral [°]'


    def hrir_positions(self, subject_id, fundamental_angles=None, orthogonal_angles=None, coordinate_system='interaural'):
        return super().hrir_positions(subject_id, coordinate_system, fundamental_angles, orthogonal_angles)


    @staticmethod
    def _convert_positions(coordinate_system, positions):
        if coordinate_system == 'cartesian':
            positions = np.stack(cartesian2interaural(*positions.T), axis=1)
        else:
            positions = np.stack(spherical2interaural(*positions.T), axis=1)
        positions[:, [0, 1]] = positions[:, [1, 0]]
        return positions


    @staticmethod
    def _coordinate_transform(coordinate_system, selected_fundamental_angles, selected_orthogonal_angles, selected_radii):
        if coordinate_system == 'interaural':
            coordinates = selected_fundamental_angles, selected_orthogonal_angles, selected_radii
        elif coordinate_system == 'spherical':
            coordinates = interaural2spherical(selected_orthogonal_angles, selected_fundamental_angles, selected_radii)
            coordinates[0], coordinates[1] = coordinates[1], coordinates[0]
        elif coordinate_system == 'cartesian':
            coordinates = interaural2cartesian(selected_fundamental_angles, selected_fundamental_angles, selected_radii)
            coordinates[0], coordinates[1] = coordinates[1], coordinates[0]
        else:
            raise ValueError(f'Unknown coordinate system "{coordinate_system}"')
        return coordinates


    @staticmethod
    def _mirror_hrirs(hrirs, _):
        # flip lateral angles (in columns)
        return np.fliplr(hrirs)


    @staticmethod
    def _verify_angle_symmetry(_, orthogonal_angles):
        # mirror laterals/columns
        if not np.allclose(orthogonal_angles, -np.flip(orthogonal_angles)):
            raise ValueError('Only datasets with symmetric lateral angles can mix mirrored and non-mirrored sides.')


class AnthropometryDataReader(DataReader):

    def anthropometric_data(self, subject_id, side, select=None):
        subject_idx = np.squeeze(np.argwhere(np.squeeze(self.query._anthropometric_ids) == subject_id))
        if subject_idx.size == 0:
            raise ValueError(f'Subject id "{subject_id}" has no anthropometric measurements')
        selected_data = self.query._anthropometry_values(side, select)[subject_idx]
        if np.all(np.isnan(selected_data), axis=-1):
            raise ValueError(f'Subject id "{subject_id}" has no data available for selection "{", ".join(select) if select is not None else "None"}"')
        return selected_data
    

class ImageDataReader(DataReader):

    @abstractmethod
    def _image_path(self, subject_id, side, rear=False):
        pass


    def image(self, subject_id, side, rear=False):
        img = Image.open(self._image_path(subject_id, side, rear))
        if side.startswith('mirrored-'):
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class CipicDataReader(SofaInterauralDataReader, AnthropometryDataReader, ImageDataReader):

    def __init__(self,
        sofa_directory_path: Union[str, Path] = '',
        image_directory_path: Union[str, Path] = '',
        anthropometry_matfile_path: Union[str, Path] = '',
        download: bool = False,
        verify: bool = True,
    ):
        query = CipicDataQuery(sofa_directory_path, image_directory_path, anthropometry_matfile_path, download, verify)
        super().__init__(query)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / f'subject_{subject_id:03d}.sofa')


    def _image_path(self, subject_id, side, rear=False):
        candidate_paths = [self.query.image_directory_path /  f'Subject_{subject_id:03d}' / f'{subject_id:03d}{suffix}' for suffix in self.query._image_suffix(side, rear)]

        for image_path in candidate_paths:
            if image_path.exists():
                return image_path
        raise ValueError(f'No {side} {"rear" if rear else "side"} image available for subject {subject_id}')


class AriDataReader(SofaSphericalDataReader, AnthropometryDataReader):

    def __init__(self,
        sofa_directory_path: Union[str, Path] = '',
        anthropometry_matfile_path: Union[str, Path] = '',
        download: bool = False,
        verify: bool = True,
    ):
        query = AriDataQuery(sofa_directory_path, anthropometry_matfile_path, download, verify)
        super().__init__(query)


    def _sofa_path(self, subject_id):
        try:
            return str(next(self.query.sofa_directory_path.glob(f'hrtf [bc]_nh{subject_id}.sofa')))
        except :
            raise ValueError(f'No subject with id "{subject_id}" exists in the collection')


class ListenDataReader(SofaSphericalDataReader, AnthropometryDataReader):

    def __init__(self,
        sofa_directory_path: Union[str, Path] = '',
        anthropometry_directory_path: Union[str, Path] = '',
        hrir_variant: str = 'compensated',
        download: bool = False,
        verify: bool = True,
    ):
        query = ListenDataQuery(sofa_directory_path, anthropometry_directory_path, hrir_variant, download, verify)
        super().__init__(query)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / self.query._checksum_key / '44100' / f'IRC_{subject_id:04d}_{self.query._hrir_variant_char}_44100.sofa')


class CrossModDataReader(SofaSphericalDataReader):

    def __init__(self,
        sofa_directory_path: Union[str, Path] = '',
        hrir_variant: str = 'compensated',
        download: bool = False,
        verify: bool = True,
    ):
        query = CrossModDataQuery(sofa_directory_path, hrir_variant, download, verify)
        super().__init__(query)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / self.query._checksum_key / '44100' / f'IRC_{subject_id:04d}_{self.query._hrir_variant_char}_44100.sofa')


class BiLiDataReader(SofaSphericalDataReader):

    def __init__(self,
        sofa_directory_path: Union[str, Path] = '',
        hrir_variant: str = 'compensated',
        hrir_samplerate: Optional[float] = None,
        download: bool = False,
        verify: bool = True,
    ):
        query = BiLiDataQuery(sofa_directory_path, hrir_samplerate, hrir_variant, download, verify)
        super().__init__(query)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / self.query._hrir_variant / str(self.query._samplerate) / f'IRC_{subject_id:04d}_{self.query._hrir_variant_char}_HRIR_{self.query._samplerate}.sofa')


class ItaDataReader(SofaSphericalDataReader, AnthropometryDataReader):

    def __init__(self,
        sofa_directory_path: Union[str, Path] = '',
        anthropometry_csvfile_path: Union[str, Path] = '',
        download: bool = False,
        verify: bool = True,
    ):
        query = ItaDataQuery(sofa_directory_path, anthropometry_csvfile_path, download, verify)
        super().__init__(query)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / f'MRT{subject_id:02d}.sofa')


class HutubsDataReader(SofaSphericalDataReader, AnthropometryDataReader):

    def __init__(self,
        sofa_directory_path: Union[str, Path] = '',
        anthropometry_csvfile_path: Union[str, Path] = '',
        hrir_method: Optional[str] = None,
        download: bool = False,
        verify: bool = True,
    ):
        query = HutubsDataQuery(sofa_directory_path, anthropometry_csvfile_path, hrir_method, download, verify)
        super().__init__(query)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / f'pp{subject_id:d}_HRIRs_{self.query._method_str}.sofa')


class RiecDataReader(SofaSphericalDataReader):

    def __init__(self,
        sofa_directory_path: Union[str, Path] = '',
        download: bool = False,
        verify: bool = True,
    ):
        query = RiecDataQuery(sofa_directory_path, download, verify)
        super().__init__(query)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / f'RIEC_hrir_subject_{subject_id:03d}.sofa')


class ChedarDataReader(SofaSphericalDataReader, AnthropometryDataReader):

    _angle_quantisation: int = 5


    def __init__(self,
        sofa_directory_path: Union[str, Path] = '',
        anthropometry_matfile_path: Union[str, Path] = '',
        distance: Optional[Union[float, str]] = None,
        download: bool = False,
        verify: bool = True,
    ):
        query = ChedarDataQuery(sofa_directory_path, anthropometry_matfile_path, distance, download, verify)
        super().__init__(query)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / self.query._radius / f'chedar_{subject_id:04d}_UV{self.query._radius}.sofa')


class WidespreadDataReader(SofaSphericalDataReader):

    def __init__(self,
        sofa_directory_path: Union[str, Path] = '',
        distance: Optional[Union[float, str]] = None,
        grid: str = 'UV',
        download: bool = False,
        verify: bool = True,
    ):
        query = WidespreadDataQuery(sofa_directory_path, distance, grid, download, verify)
        super().__init__(query)
        if grid == 'UV':
            self._angle_quantisation = 5
        else:
            self._angle_quantisation = 0

    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / self.query._grid / self.query._radius / f'{self.query._grid}{self.query._radius}_{subject_id:05d}.sofa')


class Sadie2DataReader(SofaSphericalDataReader, ImageDataReader):

    def __init__(self,
        sofa_directory_path: Union[str, Path] = '',
        image_directory_path: Union[str, Path] = '',
        hrir_samplerate: Optional[float] = None,
        download: bool = False,
        verify: bool = True,
    ):
        query = Sadie2DataQuery(sofa_directory_path, image_directory_path, hrir_samplerate, download, verify)
        super().__init__(query)


    def _sofa_path(self, subject_id):
        if subject_id < 3:
            sadie2_id = f'D{subject_id}'
        else:
            sadie2_id = f'H{subject_id}'
        return str(self.query.sofa_directory_path / sadie2_id / f'{sadie2_id}_HRIR_SOFA' / f'{sadie2_id}_{self.query._samplerate_str}_FIR_SOFA.sofa')


    def _image_path(self, subject_id, side, rear=False):
        if rear:
            raise ValueError('No rear pictures available in the SADIE II dataset')
        side_str = self.query._image_side_str(side)
        if subject_id < 3:
            sadie2_id = f'D{subject_id}'
            side_str = ' ' + side_str
        else:
            sadie2_id = f'H{subject_id}'
            side_str = '_' + side_str
        return str(self.query.image_directory_path / sadie2_id / f'{sadie2_id}_Scans' / f'{sadie2_id}{side_str}.png')


class Princeton3D3ADataReader(SofaSphericalDataReader, AnthropometryDataReader):

    def __init__(self,
        sofa_directory_path: Union[str, Path] = '',
        anthropometry_directory_path: Union[str, Path] = '',
        hrir_method: Optional[str] = None,
        hrir_variant: str = 'compensated',
        download: bool = False,
        verify: bool = True,
    ):
        query = Princeton3D3ADataQuery(sofa_directory_path, anthropometry_directory_path, hrir_method, hrir_variant, download, verify)
        super().__init__(query)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / self.query._method_str / f'Subject{subject_id}' / f'Subject{subject_id}_{self.query._hrir_variant_str}.sofa')



class ScutDataReader(SofaSphericalDataReader, AnthropometryDataReader):

    def __init__(self,
        sofa_directory_path: Union[str, Path] = '',
        anthropometry_csvfile_path: Union[str, Path] = '',
        download: bool = False,
        verify: bool = True,
    ):
        query = ScutDataQuery(sofa_directory_path, anthropometry_csvfile_path, download, verify)
        super().__init__(query)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / f'SCUT_NF_subject{subject_id:04d}_measured.sofa')


class SonicomDataReader(SofaSphericalDataReader):

    def __init__(self,
        sofa_directory_path: Union[str, Path] = '',
        hrir_variant: str = 'compensated',
        hrir_samplerate: Optional[float] = None,
        download: bool = False,
        verify: bool = True,
    ):
        query = SonicomDataQuery(sofa_directory_path, hrir_samplerate, hrir_variant, download, verify)
        super().__init__(query)


    def _sofa_path(self, subject_id):
        if isinstance(subject_id, int):
            base_dir = f'P{subject_id:04d}'
        else:
            base_dir = subject_id
        return str(self.query.sofa_directory_path / base_dir / 'HRTF' / 'HRTF' / self.query._samplerate_str / f'{base_dir}_{self.query._hrir_variant_str}_{self.query._samplerate_str}.sofa')


class MitKemarDataReader(SofaSphericalDataReader):

    def __init__(self,
        sofa_directory_path: Union[str, Path] = '',
        download: bool = False,
        verify: bool = True,
    ):
        query = MitKemarDataQuery(sofa_directory_path, download, verify)
        super().__init__(query)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / f'mit_kemar_{subject_id}_pinna.sofa')


class CustomSphericalDataReader(SofaSphericalDataReader):

    def __init__(self,
        collection_id: str,
        file_paths: Iterable[Union[str, Path]],
    ):
        query = CustomDataQuery(collection_id, file_paths)
        super().__init__(query)


    def _sofa_path(self, subject_id):
        return str(self.query.file_paths[subject_id])
