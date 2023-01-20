from .query import DataQuery, CipicDataQuery, AriDataQuery, ListenDataQuery, BiLiDataQuery, ItaDataQuery, HutubsDataQuery, RiecDataQuery, ChedarDataQuery, WidespreadDataQuery, Sadie2DataQuery, ThreeDThreeADataQuery, SonicomDataQuery
from .util import wrap_closed_open_interval, wrap_closed_interval, spherical2cartesian, spherical2interaural, cartesian2spherical, cartesian2interaural, interaural2spherical, interaural2cartesian
from abc import abstractmethod
from typing import Optional
import numpy as np
import numpy.typing as npt
import netCDF4 as ncdf
from scipy.fft import rfft, fft, ifft, rfftfreq
from scipy.signal import hilbert
from PIL import Image
from samplerate import resample


class DataReader:

    def __init__(self,
        query: DataQuery,
        verbose: bool = False,
        dtype: npt.DTypeLike = np.float32,
    ):
        self.query = query
        self.verbose = verbose
        self.dtype = dtype


class SofaDataReader(DataReader):
    """
    An abstract class that reads the HRIR for a given subject id of a dataset from a SOFA file and stores it internally
    as a 3D tensor. The HRIR directions are stored in the rows and columns as a plate carrée projection where each
    column represents a plane parallel to the fundamental plane, i.e. each row represents a single angle in the
    fundamental plane. The poles (if present) are therefore stored in the first and last column.

    row_angle: angle in the fundamental plane with range  [-180, 180)
    (azimuth for spherical coordinates, vertical angle for interaural coordinates)
    column_angle: angle between fundamental plane and directional vector with range [-90, 90]
    (elevation for spherical coordinates, lateral angle for interaural coordinates)
    """

    _quantisation: int = 3


    def __init__(self, 
        query: DataQuery,
        resample_rate: Optional[float] = None, 
        truncate_length: Optional[int] = None,
        min_phase: bool = False,
        verbose: bool = False,
        dtype: npt.DTypeLike = np.float32,
    ):
        super().__init__(query, verbose, dtype)
        self._resample_rate = resample_rate
        self._truncate_length = truncate_length
        self._min_phase = min_phase


    @property
    @abstractmethod
    def row_angle_name(self):
        pass


    @property
    @abstractmethod
    def column_angle_name(self):
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
    def _coordinate_transform(coordinate_system, selected_row_angles, selected_column_angles, selected_radii):
        pass


    @staticmethod
    @abstractmethod
    def _mirror_hrirs(hrirs, selected_row_angles):
        pass


    @staticmethod
    @abstractmethod
    def _verify_angle_symmetry(row_angles, column_angles):
        pass


    def hrir_samplerate(self, subject_id):
        if self._resample_rate is None:
            sofa_path = self._sofa_path(subject_id)
            hrir_file = ncdf.Dataset(sofa_path)
            try:
                samplerate = hrir_file.variables['Data.SamplingRate'][:].item()
            except Exception as exc:
                raise ValueError(f'Error reading file "{sofa_path}"') from exc
            finally:
                hrir_file.close()
            return samplerate
        return self._resample_rate


    def hrir_length(self, subject_id):
        if self._truncate_length is None:
            sofa_path = self._sofa_path(subject_id)
            hrir_file = ncdf.Dataset(sofa_path)
            try:
                length = hrir_file.dimensions['N'].size
            except Exception as exc:
                raise ValueError(f'Error reading file "{sofa_path}"') from exc
            finally:
                hrir_file.close()
            return length
        return self._truncate_length


    def hrtf_frequencies(self, subject_id):
        num_samples = self.hrir_length(subject_id)
        return rfftfreq(num_samples, 1/self.hrir_samplerate(subject_id))


    def _map_sofa_position_order_to_matrix(self, subject_id, row_angles, column_angles):
        if row_angles is not None and column_angles is not None and len(row_angles) != len(column_angles):
            raise ValueError(f'The number of row angles ({len(row_angles)}) differs from the number of column angles ({len(column_angles)})')
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
        quantised_positions = np.round(positions, self._quantisation)
        quantised_positions[:, 0] = wrap_closed_open_interval(quantised_positions[:, 0], -180, 180)

        if row_angles is not None:
            row_angles = wrap_closed_open_interval(row_angles, -180, 180)
        if column_angles is not None:
            column_angles = wrap_closed_interval(column_angles, -90, 90)
        radii = np.unique(quantised_positions[:, 2])
        if row_angles is None and column_angles is None:
            # Read all positions in file, without extra checks
            selected_file_indices = list(range(len(quantised_positions)))
        else:
            # Check if file positions are part of requested positions
            selected_file_indices = []
            if row_angles is not None:
                if column_angles is not None:
                    check = lambda file_row_angle, file_column_angle: (np.isclose(file_row_angle, row_angles) & np.isclose(file_column_angle, column_angles)).any()
                else:
                    check = lambda file_row_angle, _: np.isclose(file_row_angle, row_angles).any()
            elif column_angles is not None:
                check = lambda _, file_column_angle: np.isclose(file_column_angle, column_angles).any()
            for file_idx, (file_row_angle, file_column_angle, file_radius) in enumerate(quantised_positions):
                if check(file_row_angle, file_column_angle):
                    selected_file_indices.append(file_idx)

        selected_positions = quantised_positions[selected_file_indices]
        selected_row_angles = selected_positions[:, 0]
        selected_column_angles = selected_positions[:, 1]
        selected_radii = selected_positions[:, 2]

        for pole_angle in (-90, 90):
            # If value at pole requested
            if column_angles is None or np.isclose(column_angles, pole_angle).any():
                for radius in radii:
                    pole_indices = np.flatnonzero(np.isclose(quantised_positions[:, 1], pole_angle) & (quantised_positions[:, 2] == radius))
                    # If at least one value at pole present in file
                    if len(pole_indices) > 0:
                        # Make sure to include all requested row angles at pole
                        if row_angles is not None and column_angles is not None:
                            requested_row_angles_at_pole = row_angles[column_angles == pole_angle]
                            selected_row_angles = np.concatenate((selected_row_angles, requested_row_angles_at_pole))
                            selected_radii = np.concatenate((selected_radii, [radius]))
                        # If pole angle not present in selection yet
                        if len(selected_column_angles) == 0 or not np.isclose(selected_column_angles, pole_angle).any():
                            # Add to column angles at appropriate extremum
                            if pole_angle > 0 or len(selected_column_angles) == 0:
                                selected_column_angles = np.append(selected_column_angles, pole_angle)
                            else:
                                selected_column_angles = np.insert(selected_column_angles, 0, pole_angle)

        unique_row_angles = np.unique(selected_row_angles)
        unique_column_angles = np.unique(selected_column_angles)
        unique_radii = np.unique(selected_radii)

        additional_pole_indices = []

        def repeat_value_at_pole(pole_angle, pole_column_idx):
            # If value at pole requested
            if column_angles is None or np.isclose(column_angles, pole_angle).any():
                for radius_idx, radius in enumerate(radii):
                    pole_indices = np.flatnonzero(np.isclose(quantised_positions[:, 1], pole_angle) & (quantised_positions[:, 2] == radius))
                    # If at least one value at pole present in file
                    if len(pole_indices) > 0:
                        pole_idx = pole_indices[0]
                        # Copy to those row angles that miss the value at the pole (if any)
                        radius_positions = selected_positions[selected_positions[:, 2] == radius]
                        present_row_angles = radius_positions[np.isclose(radius_positions[:, 1], pole_angle), 0]
                        missing_row_angles = np.setdiff1d(unique_row_angles, present_row_angles)
                        missing_row_indices = [np.argmax(angle == unique_row_angles) for angle in missing_row_angles]
                        selected_file_indices.extend([pole_idx] * len(missing_row_indices))
                        additional_pole_indices.extend([(row_idx, pole_column_idx, radius_idx) for row_idx in missing_row_indices])

        repeat_value_at_pole(-90, 0)
        repeat_value_at_pole(90, -1)

        selection_mask_indices = []
        for file_row_angle, file_column_angle, file_radius in selected_positions:
            selection_mask_indices.append((np.argmax(file_row_angle == unique_row_angles), np.argmax(file_column_angle == unique_column_angles), np.argmax(file_radius == unique_radii)))
        selection_mask_indices.extend(additional_pole_indices)
        if len(selection_mask_indices) == 0:
            raise ValueError('None of the specified angles are available in this dataset')

        selection_mask_indices = tuple(np.array(selection_mask_indices).T)
        selection_mask = np.full((len(unique_row_angles), len(unique_column_angles), len(unique_radii)), True)
        selection_mask[selection_mask_indices] = False

        return unique_row_angles, unique_column_angles, unique_radii, selection_mask, selected_file_indices, selection_mask_indices


    def hrir_positions(self, subject_id, coordinate_system, row_angles=None, column_angles=None):
        selected_row_angles, selected_column_angles, selected_radii, selection_mask, *_ = self._map_sofa_position_order_to_matrix(subject_id, row_angles, column_angles)

        coordinates = self._coordinate_transform(coordinate_system, selected_row_angles, selected_column_angles, selected_radii)

        position_grid = np.stack(np.meshgrid(*coordinates, indexing='ij'), axis=-1)
        if selection_mask.any(): # sparse grid
            tiled_position_mask = np.tile(selection_mask[:, :, :, np.newaxis], (1, 1, 1, 3))
            return np.ma.masked_where(tiled_position_mask, position_grid)
        # dense grid
        return position_grid


    def hrir(self, subject_id, side, domain='time', row_angles=None, column_angles=None):
        sofa_path = self._sofa_path(subject_id)
        hrir_file = ncdf.Dataset(sofa_path)
        try:
            hrirs = np.ma.getdata(hrir_file.variables['Data.IR'][:, 0 if side.endswith('left') else 1, :])
            samplerate = hrir_file.variables['Data.SamplingRate'][:].item()
        except Exception as exc:
            raise ValueError(f'Error reading file "{sofa_path}"') from exc
        finally:
            hrir_file.close()
        selected_row_angles, _, _, selection_mask, selected_file_indices, selection_mask_indices = self._map_sofa_position_order_to_matrix(subject_id, row_angles, column_angles)
        hrir_matrix = np.empty(selection_mask.shape + (hrirs.shape[1],))
        hrir_matrix[selection_mask_indices] = hrirs[selected_file_indices]

        if self._min_phase:
            hrtf_matrix = fft(hrir_matrix, int(samplerate))
            magnitudes = np.abs(hrtf_matrix)
            min_phases = -np.imag(hilbert(np.log(magnitudes)))
            min_phase_hrtf_matrix = magnitudes * np.exp(1j * min_phases)
            hrir_matrix = np.real(ifft(min_phase_hrtf_matrix, int(samplerate))[:, :, :, :hrir_matrix.shape[-1]])
        if self._resample_rate is not None and self._resample_rate != samplerate:
            channel_view = hrir_matrix.reshape(-1, hrir_matrix.shape[-1])
            hrir_matrix = np.hstack([resample(channel_view[ch_idx:ch_idx+128].T, self._resample_rate / samplerate) for ch_idx in range(0, len(channel_view), 128)]).T.reshape(*hrir_matrix.shape[:3], -1)
        if self._truncate_length is not None:
            hrir_matrix = hrir_matrix[:, :, :, :self._truncate_length]
        tiled_position_mask = np.tile(selection_mask[:, :, :, np.newaxis], (1, 1, 1, hrir_matrix.shape[-1]))
        if domain == 'time':
            hrir = np.ma.masked_where(tiled_position_mask, hrir_matrix, copy=False)
        else:
            hrtf_matrix = np.ma.masked_where(tiled_position_mask[:, :, :, :hrir_matrix.shape[-1]//2+1], rfft(hrir_matrix), copy=False)
            if domain == 'complex':
                hrir = hrtf_matrix
            elif domain.startswith('magnitude'):
                magnitudes = np.abs(hrtf_matrix)
                if domain.endswith('_db'):
                    # limit dB range to what is representable in data type
                    min_magnitude = np.max(magnitudes) * np.finfo(self.dtype).resolution
                    hrir = 20*np.log10(np.clip(magnitudes, min_magnitude, None))
                else:
                    hrir = magnitudes
            elif domain == 'phase':
                hrir = np.angle(hrtf_matrix)
            else:
                raise ValueError(f'Unknown domain "{domain}" for HRIR')
        if domain == 'complex' and not issubclass(self.dtype, np.complexfloating):
            raise ValueError(f'An HRTF in the complex domain requires the dtype to be set to a complex type (currently {self.dtype})')
        hrir = hrir.astype(self.dtype)
        if side.startswith('mirrored'):
            return self._mirror_hrirs(hrir, selected_row_angles)
        return hrir


class SofaSphericalDataReader(SofaDataReader):

    @property
    def row_angle_name(self):
        return 'azimuth [°]'


    @property
    def column_angle_name(self):
        return 'elevation [°]'


    def hrir_positions(self, subject_id, row_angles=None, column_angles=None, coordinate_system='spherical'):
        return super().hrir_positions(subject_id, coordinate_system, row_angles, column_angles)


    @staticmethod
    def _convert_positions(coordinate_system, positions):
        if coordinate_system == 'cartesian':
            positions = np.stack(cartesian2spherical(*positions.T), axis=1)
        return positions


    @staticmethod
    def _coordinate_transform(coordinate_system, selected_row_angles, selected_column_angles, selected_radii):
        if coordinate_system == 'spherical':
            return selected_row_angles, selected_column_angles, selected_radii
        if coordinate_system == 'interaural':
            return spherical2interaural(selected_row_angles, selected_column_angles, selected_radii)
        if coordinate_system == 'cartesian':
            return spherical2cartesian(selected_row_angles, selected_column_angles, selected_radii)
        raise ValueError(f'Unknown coordinate system "{coordinate_system}"')


    @staticmethod
    def _mirror_hrirs(hrirs, selected_row_angles):
        # flip azimuths (in rows)
        if np.isclose(selected_row_angles[0], -180):
            return np.ma.row_stack((hrirs[0:1], np.flipud(hrirs[1:])))
        else:
            return np.flipud(hrirs)


    @staticmethod
    def _verify_angle_symmetry(row_angles, _):
        # mirror azimuths/rows
        start_idx = 1 if np.isclose(row_angles[0], -180) else 0
        if not np.allclose(row_angles[start_idx:], -np.flip(row_angles[start_idx:])):
            raise ValueError('Only datasets with symmetric azimuths can mix mirrored and non-mirrored sides.')


class SofaInterauralDataReader(SofaDataReader):

    @property
    def row_angle_name(self):
        return 'vertical [°]'


    @property
    def column_angle_name(self):
        return 'lateral [°]'


    def hrir_positions(self, subject_id, row_angles=None, column_angles=None, coordinate_system='interaural'):
        return super().hrir_positions(subject_id, coordinate_system, row_angles, column_angles)


    @staticmethod
    def _convert_positions(coordinate_system, positions):
        if coordinate_system == 'cartesian':
            positions = np.stack(cartesian2interaural(*positions.T), axis=1)
        else:
            positions = np.stack(spherical2interaural(*positions.T), axis=1)
        positions[:, [0, 1]] = positions[:, [1, 0]]
        return positions


    @staticmethod
    def _coordinate_transform(coordinate_system, selected_row_angles, selected_column_angles, selected_radii):
        if coordinate_system == 'interaural':
            coordinates = selected_row_angles, selected_column_angles, selected_radii
        elif coordinate_system == 'spherical':
            coordinates = interaural2spherical(selected_column_angles, selected_row_angles, selected_radii)
            coordinates[0], coordinates[1] = coordinates[1], coordinates[0]
        elif coordinate_system == 'cartesian':
            coordinates = interaural2cartesian(selected_row_angles, selected_row_angles, selected_radii)
            coordinates[0], coordinates[1] = coordinates[1], coordinates[0]
        else:
            raise ValueError(f'Unknown coordinate system "{coordinate_system}"')
        return coordinates


    @staticmethod
    def _mirror_hrirs(hrirs, _):
        # flip lateral angles (in columns)
        return np.fliplr(hrirs)


    @staticmethod
    def _verify_angle_symmetry(_, column_angles):
        # mirror laterals/columns
        if not np.allclose(column_angles, -np.flip(column_angles)):
            raise ValueError('Only datasets with symmetric lateral angles can mix mirrored and non-mirrored sides.')


class MatFileAnthropometryDataReader(DataReader):

    def anthropomorphic_data(self, subject_id, side=None, select=None):
        select_all = ('head-torso', 'pinna-size', 'pinna-angle', 'weight', 'age', 'sex')
        if select is None:
            select = select_all
        elif isinstance(select, str):
            select = (select,)
        # if 'pinna-size' not in select and 'pinna-angle' not in select:
        #     if side is not None:
        #         print(f'Side "{side}" is irrelevant for this measurements selection "{", ".join(select)}"')
        # el
        if side not in ['left', 'right', 'both']: # and ('pinna-size' in select or 'pinna-angle' in select)
            raise ValueError(f'Unknown side selector "{side}"')

        unknown_select = sorted(set(select) - set(select_all))
        if unknown_select:
            raise ValueError(f'Unknown selection "{unknown_select}". Choose one or more from "{select_all}"')

        subject_idx = np.squeeze(np.argwhere(np.squeeze(self.anth['id']) == subject_id))
        if subject_idx.size == 0:
            raise ValueError(f'Subject id "{subject_id}" has no anthropomorphic measurements')

        select_data = []

        if 'head-torso' in select:
            select_data.append(self.anth['X'][subject_idx])
        if side == 'left' or side.startswith('both'):
            if 'pinna-size' in select:
                select_data.append(self.anth['D'][subject_idx, :8])
            if 'pinna-angle' in select:
                select_data.append(self.anth['theta'][subject_idx, :2])
        if side == 'right' or side.startswith('both'):
            if 'pinna-size' in select:
                select_data.append(self.anth['D'][subject_idx, 8:])
            if 'pinna-angle' in select:
                select_data.append(self.anth['theta'][subject_idx, 2:])
        if 'weight' in select:
            select_data.append(self.anth['WeightKilograms'][subject_idx])
        if 'age' in select:
            select_data.append(self.anth['age'][subject_idx])
        if 'sex' in select:
            select_data.append(0 if self.anth['sex'][subject_idx] == 'M' else 1 if self.anth['sex'][subject_idx] == 'F' else np.nan)

        selected_data = np.hstack(select_data).astype(self.dtype)
        if np.all(np.isnan(selected_data), axis=-1):
            raise ValueError(f'Subject id "{subject_id}" has no data available for selection "{", ".join(select)}"')
        return selected_data


class ImageDataReader(DataReader):

    @abstractmethod
    def _image_path(self, subject_id, side=None, rear=False):
        pass


    def image(self, subject_id, side=None, rear=False):
        img = Image.open(self.pinna_image_path(subject_id, side, rear))
        if side.startwith('mirrored-'):
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class CipicDataReader(SofaInterauralDataReader, ImageDataReader, MatFileAnthropometryDataReader):

    def __init__(self,
        sofa_directory_path: str = None,
        image_directory_path: str = None,
        anthropomorphy_matfile_path: str = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[int] = None,
        hrir_min_phase: bool = False,
        verbose: bool = False,
        dtype: npt.DTypeLike = np.float32,
    ):
        query = CipicDataQuery(sofa_directory_path, image_directory_path, anthropomorphy_matfile_path)
        super().__init__(query, hrir_samplerate, hrir_length, hrir_min_phase, verbose, dtype)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / 'subject_{:03d}.sofa'.format(subject_id))


class AriDataReader(SofaSphericalDataReader, MatFileAnthropometryDataReader):

    def __init__(self,
        sofa_directory_path: str = None,
        anthropomorphy_matfile_path: str = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[int] = None,
        hrir_min_phase: bool = False,
        verbose: bool = False,
        dtype: npt.DTypeLike = np.float32,
    ):
        query = AriDataQuery(sofa_directory_path, anthropomorphy_matfile_path)
        super().__init__(query, hrir_samplerate, hrir_length, hrir_min_phase, verbose, dtype)


    def _sofa_path(self, subject_id):
        try:
            return str(next(self.query.sofa_directory_path.glob('hrtf [bc]_nh{}.sofa'.format(subject_id))))
        except :
            raise ValueError(f'No subject with id "{subject_id}" exists in the collection')


class ListenDataReader(SofaSphericalDataReader):

    def __init__(self,
        sofa_directory_path: str = None,
        hrtf_type: str = 'compensated',
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[int] = None,
        hrir_min_phase: bool = False,
        verbose: bool = False,
        dtype: npt.DTypeLike = np.float32,
    ):
        query = ListenDataQuery(sofa_directory_path, hrtf_type)
        super().__init__(query, hrir_samplerate, hrir_length, hrir_min_phase, verbose, dtype)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / 'IRC_{:04d}_{}_44100.sofa'.format(subject_id, self.query._hrtf_type_str))


class BiLiDataReader(SofaSphericalDataReader):

    def __init__(self,
        sofa_directory_path: str = None,
        hrtf_type: str = 'compensated',
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[int] = None,
        hrir_min_phase: bool = False,
        verbose: bool = False,
        dtype: npt.DTypeLike = np.float32,
    ):
        query = BiLiDataQuery(sofa_directory_path, hrir_samplerate, hrtf_type)
        super().__init__(query, hrir_samplerate, hrir_length, hrir_min_phase, verbose, dtype)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / 'IRC_{:04d}_{}_HRIR_{}.sofa'.format(subject_id, self.query._hrtf_type_str, self.query._samplerate))


class ItaDataReader(SofaSphericalDataReader):

    def __init__(self,
        sofa_directory_path: str = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[int] = None,
        hrir_min_phase: bool = False,
        verbose: bool = False,
        dtype: npt.DTypeLike = np.float32,
    ):
        query = ItaDataQuery(sofa_directory_path)
        super().__init__(query, hrir_samplerate, hrir_length, hrir_min_phase, verbose, dtype)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / 'MRT{:02d}.sofa'.format(subject_id))


class HutubsDataReader(SofaSphericalDataReader):

    def __init__(self,
        sofa_directory_path: str = None,
        measured_hrtf: bool = True,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[int] = None,
        hrir_min_phase: bool = False,
        verbose: bool = False,
        dtype: npt.DTypeLike = np.float32,
    ):
        query = HutubsDataQuery(sofa_directory_path, measured_hrtf)
        super().__init__(query, hrir_samplerate, hrir_length, hrir_min_phase, verbose, dtype)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / 'pp{:d}_HRIRs_{}.sofa'.format(subject_id, self.query._variant_key))


class RiecDataReader(SofaSphericalDataReader):

    def __init__(self,
        sofa_directory_path: str = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[int] = None,
        hrir_min_phase: bool = False,
        verbose: bool = False,
        dtype: npt.DTypeLike = np.float32,
    ):
        query = RiecDataQuery(sofa_directory_path)
        super().__init__(query, hrir_samplerate, hrir_length, hrir_min_phase, verbose, dtype)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / f'RIEC_hrir_subject_{subject_id:03d}.sofa')


class ChedarDataReader(SofaSphericalDataReader):

    def __init__(self,
        sofa_directory_path: str = None,
        radius: float = 1,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[int] = None,
        hrir_min_phase: bool = False,
        verbose: bool = False,
        dtype: npt.DTypeLike = np.float32,
    ):
        query = ChedarDataQuery(sofa_directory_path, radius)
        super().__init__(query, hrir_samplerate, hrir_length, hrir_min_phase, verbose, dtype)
        self._quantisation = 1


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / f'chedar_{subject_id:04d}_UV{self.query._radius}.sofa')


class WidespreadDataReader(SofaSphericalDataReader):

    def __init__(self,
        sofa_directory_path: str = None,
        radius: float = 1,
        grid: str = 'UV',
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[int] = None,
        hrir_min_phase: bool = False,
        verbose: bool = False,
        dtype: npt.DTypeLike = np.float32,
    ):
        query = WidespreadDataQuery(sofa_directory_path, radius, grid)
        super().__init__(query, hrir_samplerate, hrir_length, hrir_min_phase, verbose, dtype)
        self._quantisation = 1


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / f'{self.query._grid}{self.query._radius}_{subject_id:05d}.sofa')


class Sadie2DataReader(SofaSphericalDataReader, ImageDataReader):

    def __init__(self,
        sofa_directory_path: str = None,
        image_directory_path: str = None,
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[int] = None,
        hrir_min_phase: bool = False,
        verbose: bool = False,
        dtype: npt.DTypeLike = np.float32,
    ):
        query = Sadie2DataQuery(sofa_directory_path, image_directory_path, hrir_samplerate)
        super().__init__(query, hrir_samplerate, hrir_length, hrir_min_phase, verbose, dtype)


    def _sofa_path(self, subject_id):
        if subject_id < 3:
            sadie2_id = f'D{subject_id}'
        else:
            sadie2_id = f'H{subject_id}'
        return str(self.query.sofa_directory_path / f'{sadie2_id}/{sadie2_id}_HRIR_SOFA/{sadie2_id}_{self.query._samplerate_str}_FIR_SOFA.sofa')


class ThreeDThreeADataReader(SofaSphericalDataReader):

    def __init__(self,
        sofa_directory_path: str = None,
        hrtf_method: str = 'measured',
        hrtf_type: str = 'compensated',
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[int] = None,
        hrir_min_phase: bool = False,
        verbose: bool = False,
        dtype: npt.DTypeLike = np.float32,
    ):
        query = ThreeDThreeADataQuery(sofa_directory_path, hrtf_method, hrtf_type)
        super().__init__(query, hrir_samplerate, hrir_length, hrir_min_phase, verbose, dtype)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / f'{self.query._method_str}/Subject{subject_id}/Subject{subject_id}_{self.query._hrtf_type_str}.sofa')


class SonicomDataReader(SofaSphericalDataReader):

    def __init__(self,
        sofa_directory_path: str = None,
        hrtf_type: str = 'compensated',
        hrir_samplerate: Optional[float] = None,
        hrir_length: Optional[int] = None,
        hrir_min_phase: bool = False,
        verbose: bool = False,
        dtype: npt.DTypeLike = np.float32,
    ):
        query = SonicomDataQuery(sofa_directory_path, hrir_samplerate, hrtf_type)
        super().__init__(query, hrir_samplerate, hrir_length, hrir_min_phase, verbose, dtype)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / f'P{subject_id:04d}/HRTF/{self.query._samplerate_str}/P{subject_id:04d}_{self.query._hrtf_type_str}_{self.query._samplerate_str}.sofa')
