from .query import DataQuery, AriDataQuery, ListenDataQuery, BiLiDataQuery, ItaDataQuery, HutubsDataQuery, RiecDataQuery, ChedarDataQuery, WidespreadDataQuery, Sadie2DataQuery, ThreeDThreeADataQuery
from .util import wrap_open_closed_interval, spherical2cartesian, spherical2interaural
from abc import abstractmethod
from pathlib import Path
import numpy as np
import sofa
from scipy.fft import rfft, fftfreq
from PIL import Image


class DataPoint:

    def __init__(self, query, dataset_id, verbose=False, dtype=np.float32):
        self.query = query
        self.dataset_id = dataset_id
        self.verbose = verbose
        self.dtype = dtype


class SofaDataPoint(DataPoint):

    @abstractmethod
    def _sofa_path(self, subject_id):
        pass
    

    def hrir_samplerate(self, subject_id):
        try:
            hrir_file = sofa.Database.open(self._sofa_path(subject_id))
            samplerate = hrir_file.Data.SamplingRate.get_values({'M': 0}).item()
        except:
            raise ValueError(f'Error reading file "{self._sofa_path(subject_id)}"')
        hrir_file.close()
        return samplerate


    def hrir_length(self, subject_id):
        hrir_file = sofa.Database.open(self._sofa_path(subject_id))
        try:
            length = hrir_file.Dimensions.N
        except:
            raise ValueError(f'Error reading file "{self._sofa_path(subject_id)}"')
        hrir_file.close()
        return length


    def hrtf_frequencies(self, subject_id):
        num_samples = self.hrir_length(subject_id)
        num_bins = num_samples // 2 + 1
        return np.abs(fftfreq(num_samples, 1./self.hrir_samplerate(subject_id))[:num_bins])


    @staticmethod
    def _hrir_select_angles(row_angles, column_angles, all_row_angles, all_column_angles, position_mask):
        if row_angles is None:
            select_row_indices = np.full(len(all_row_angles), True)
        else:
            select_row_indices = np.array([np.isclose(angle, row_angles).any() for angle in all_row_angles])
            if not any(select_row_indices):
                raise ValueError('None of the specified angles are available in this dataset')

        if column_angles is None:
            select_column_indices = np.full(len(all_column_angles), True)
        else:
            select_column_indices = np.array([np.isclose(angle, column_angles).any() for angle in all_column_angles])
            if not any(select_column_indices):
                raise ValueError('None of the specified angles are available in this dataset')

        selected_position_mask = position_mask[select_row_indices][:, select_column_indices]
        # prune those elevations that no longer have a single azimuth in the current selection and the other way around
        keep_row_indices = ~selected_position_mask.all(axis=(1,2))
        keep_column_indices = ~selected_position_mask.all(axis=(0,2))
        row_indices = select_row_indices.nonzero()[0][keep_row_indices]
        column_indices = select_column_indices.nonzero()[0][keep_column_indices]

        return row_indices, column_indices


    def _map_sofa_position_order_to_matrix(self, subject_id):
        hrir_file = sofa.Database.open(self._sofa_path(subject_id))
        try:
            positions = hrir_file.Source.Position.get_global_values(system='spherical')
        except:
            raise ValueError(f'Error reading file "{self._sofa_path(subject_id)}"')
        hrir_file.close()
        quantified_positions = np.round(positions, 2)
        quantified_positions[:, 0] = wrap_open_closed_interval(quantified_positions[:, 0], -180, 180)
        unique_azimuths = np.unique(quantified_positions[:, 0])
        unique_elevations = np.unique(quantified_positions[:, 1])
        unique_radii = np.unique(quantified_positions[:, 2])
        position_map = np.empty((3, len(positions)), dtype=int)
        for idx, (azimuth, elevation, radius) in enumerate(quantified_positions):
            position_map[:, idx] = np.argmax(azimuth == unique_azimuths), np.argmax(elevation == unique_elevations), np.argmax(radius == unique_radii)
        return unique_azimuths, unique_elevations, unique_radii, tuple(position_map)


    def hrir_angle_indices(self, subject_id, row_angles=None, column_angles=None):
        unique_row_angles, unique_column_angles, unique_radii, position_map = self._map_sofa_position_order_to_matrix(subject_id)
        position_mask = np.full((len(unique_row_angles), len(unique_column_angles), len(unique_radii)), True)
        position_mask[position_map] = False
        row_indices, column_indices = SofaDataPoint._hrir_select_angles(row_angles, column_angles, unique_row_angles, unique_column_angles, position_mask)
        selected_angles = {unique_row_angles[row_idx]: np.ma.array(unique_column_angles[column_indices], mask=position_mask[row_idx, column_indices]) for row_idx in row_indices}
        return selected_angles, row_indices, column_indices


    def hrir_positions(self, subject_id, row_angles=None, column_angles=None, coordinate_system='spherical'):
        selected_azimuths, selected_elevations, row_indices, column_indices = self.hrir_angle_indices(subject_id, row_angles, column_angles)
        selected_position_mask = position_mask[row_indices][:, column_indices]

        if coordinate_system == 'spherical':
            coordinates = selected_azimuths, selected_elevations, all_radii
        elif coordinate_system == 'interaural':
            coordinates = spherical2interaural(selected_azimuths, selected_elevations, all_radii)
        elif coordinate_system == 'cartesian':
            coordinates = spherical2cartesian(selected_azimuths, selected_elevations, all_radii)
        else:
            raise ValueError(f'Unknown coordinate system "{coordinate_system}"')
        position_grid = np.stack(np.meshgrid(*coordinates, indexing='ij'), axis=-1)
        if selected_position_mask.any(): # sparse grid
            tiled_position_mask = np.tile(selected_position_mask[:, :, :, np.newaxis], (1,1,1,3))
            return np.ma.masked_where(tiled_position_mask, position_grid)
        # dense grid
        return position_grid


    def hrir(self, subject_id, side, domain='time', row_indices=None, column_indices=None):
        hrir_file = sofa.Database.open(self._sofa_path(subject_id))
        try:
            hrirs = hrir_file.Data.IR.get_values({'R': 0 if side.endswith('left') else 1})
        except:
            raise ValueError(f'Error reading file "{self._sofa_path(subject_id)}"')
        hrir_file.close()
        unique_azimuths, unique_elevations, unique_radii, position_map = self._map_sofa_position_order_to_matrix(subject_id)
        hrir_matrix = np.empty((len(unique_azimuths), len(unique_elevations), len(unique_radii), hrirs.shape[1]))
        hrir_matrix[position_map] = hrirs
        position_mask = np.full_like(hrir_matrix, True, dtype=bool)
        position_mask[position_map] = False

        if row_indices is None:
            row_indices = slice(None)
        if column_indices is None:
            column_indices = slice(None)

        selected_position_mask = position_mask[row_indices][:, column_indices]
        selected_hrir_matrix = np.ma.masked_where(selected_position_mask, hrir_matrix[row_indices][:, column_indices], copy=False)
        if domain == 'time':
            hrir = selected_hrir_matrix.astype(self.dtype)
        else:
            selected_hrtf_matrix = np.ma.masked_where(selected_position_mask[:, :, :, :hrirs.shape[1]//2+1], rfft(selected_hrir_matrix), copy=False)
            if domain == 'complex':
                hrir = selected_hrtf_matrix.astype(self.dtype)
            elif domain.startswith('magnitude'):
                magnitudes = np.abs(selected_hrtf_matrix)
                if domain.endswith('_db'):
                    # limit dB range to what is representable in data type
                    min_magnitude = np.max(magnitudes) * np.finfo(self.dtype).resolution
                    hrir = 20*np.log10(np.clip(magnitudes, min_magnitude, None)).astype(self.dtype)
                else:
                    hrir = magnitudes
            elif domain == 'phase':
                hrir = np.angle(selected_hrtf_matrix).astype(self.dtype)
            else:
                hrir = ValueError(f'Unknown domain "{domain}" for HRIR')
        if side.startswith('flipped'):
            return np.flipud(hrir)
        return hrir


class MatFileAnthropometryDataPoint(DataPoint):

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


class ImageDataPoint(DataPoint):

    @abstractmethod
    def _image_path(self, subject_id, side=None, rear=False):
        pass


    def image(self, subject_id, side=None, rear=False):
        img = Image.open(self.pinna_image_path(subject_id, side, rear))
        if side.startwith('flipped-'):
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class AriDataPoint(SofaDataPoint, MatFileAnthropometryDataPoint):

    def __init__(
        self,
        sofa_directory_path=None,
        anthropomorphy_matfile_path=None,
        verbose=False,
        dtype=np.float32
    ):
        query = AriDataQuery(sofa_directory_path, anthropomorphy_matfile_path)
        super().__init__(query, 'ari', verbose, dtype)


    def _sofa_path(self, subject_id):
        return str(next(self.query.sofa_directory_path.glob('hrtf [bc]_nh{}.sofa'.format(subject_id))))


class ListenDataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, verbose=False, dtype=np.float32):
        query = ListenDataQuery(sofa_directory_path)
        super().__init__(query, 'listen', verbose, dtype)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / 'IRC_{:04d}_C_44100.sofa'.format(subject_id))


class BiLiDataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, verbose=False, dtype=np.float32):
        query = BiLiDataQuery(sofa_directory_path)
        super().__init__(query, 'bili', verbose, dtype)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / 'IRC_{:04d}_C_HRIR_96000.sofa'.format(subject_id))


class ItaDataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, verbose=False, dtype=np.float32):
        query = ItaDataQuery(sofa_directory_path)
        super().__init__(query, 'ita', verbose, dtype)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / 'MRT{:02d}.sofa'.format(subject_id))


class HutubsDataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, verbose=False, dtype=np.float32):
        query = HutubsDataQuery(sofa_directory_path)
        super().__init__(query, 'hutubs', verbose, dtype)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / 'pp{:d}_HRIRs_measured.sofa'.format(subject_id))


class RiecDataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, verbose=False, dtype=np.float32):
        query = RiecDataQuery(sofa_directory_path)
        super().__init__(query, 'riec', verbose, dtype)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / f'RIEC_hrir_subject_{subject_id:03d}.sofa')


class ChedarDataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, radius=1, verbose=False, dtype=np.float32):
        query = ChedarDataQuery(sofa_directory_path)
        super().__init__(query, 'chedar', verbose, dtype)
        if np.isclose(radius, 0.2):
            self.radius = '02m'
        elif np.isclose(radius, 0.5):
            self.radius = '05m'
        elif np.isclose(radius, 1):
            self.radius = '1m'
        elif np.isclose(radius, 2):
            self.radius = '2m'
        else:
            raise ValueError('The radius needs to be one of 0.2, 0.5, 1 or 2')


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / f'chedar_{subject_id:04d}_UV{self.radius}.sofa')


class WidespreadDataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, radius=1, verbose=False, dtype=np.float32):
        query = WidespreadDataQuery(sofa_directory_path)
        super().__init__(query, 'widespread', verbose, dtype)
        if np.isclose(radius, 0.2):
            self.radius = '02m'
        elif np.isclose(radius, 0.5):
            self.radius = '05m'
        elif np.isclose(radius, 1):
            self.radius = '1m'
        elif np.isclose(radius, 2):
            self.radius = '2m'
        else:
            raise ValueError('The radius needs to be one of 0.2, 0.5, 1 or 2')


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / f'UV{self.radius}_{subject_id:05d}.sofa')


class Sadie2DataPoint(SofaDataPoint, ImageDataPoint):

    def __init__(self, sofa_directory_path=None, image_directory_path=None, verbose=False, dtype=np.float32):
        query = Sadie2DataQuery(sofa_directory_path, image_directory_path)
        super().__init__(query, 'sadie2', verbose, dtype)


    def _sofa_path(self, subject_id):
        if subject_id < 3:
            sadie2_id = f'D{subject_id}'
        else:
            sadie2_id = f'H{subject_id}'
        return str(self.query.sofa_directory_path / f'{sadie2_id}/{sadie2_id}_HRIR_SOFA/{sadie2_id}_96K_24bit_512tap_FIR_SOFA.sofa')


class ThreeDThreeADataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, verbose=False, dtype=np.float32):
        query = ThreeDThreeADataQuery(sofa_directory_path)
        super().__init__(query, '3d3a', verbose, dtype)


    def _sofa_path(self, subject_id):
        return str(self.query.sofa_directory_path / f'Subject{subject_id}_HRIRs.sofa')
