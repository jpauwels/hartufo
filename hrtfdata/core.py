from .util import wrap_open_closed_interval, spherical2cartesian, spherical2interaural
from abc import abstractmethod
from pathlib import Path
import numpy as np
import sofa
from scipy import io
from scipy.fft import rfft, fftfreq


class DataPoint:

    def __init__(self, verbose=False, dtype=np.float32):
        self.verbose = verbose
        self.dtype = dtype
        self.allowed_keys = ['subject', 'side', 'dataset']


    def validate_specifications(self, feature_spec, label_spec):
        unknown_features = sorted(set(feature_spec.keys()).difference(self.allowed_keys))
        if unknown_features:
            raise ValueError(f'Unknown feature specifier{"s" if len(unknown_features) > 1 else ""} "{", ".join(unknown_features)}"')
        unknown_labels = sorted(set(label_spec.keys()).difference(self.allowed_keys))
        if unknown_labels:
            raise ValueError(f'Unknown label specifier{"s" if len(unknown_labels) > 1 else ""} "{", ".join(unknown_labels)}"')


    def specification_based_ids(self, specifications):
        if 'images' in specifications.keys() and 'measurements' in specifications.keys():
            # image_side = specifications['images'].pop('side')
            # measurements_side = specifications['measurements'].pop('side')
            # kw_args = {'image_side': image_side, **specifications['images'], 'measurements_side': measurements_side, **specifications['measurements']}
            subject_ids = self.image_measurements_ids(**{**specifications['images'], **specifications['measurements']})
        elif 'images' in specifications.keys():
            subject_ids = self.image_ids(**specifications['images'])
        elif 'measurements' in specifications.keys():
            subject_ids = self.measurements_ids(**specifications['measurements'])
        else:
            side = specifications['hrirs'].get('side', 'both')
            return self.hrir_ids(side=side)
        return sorted(set(self.hrir_ids()).intersection(subject_ids))


class SofaDataPoint(DataPoint):

    def __init__(self, sofa_directory_path, verbose=False, dtype=np.float32):
        super().__init__(verbose, dtype)
        self.sofa_directory_path = Path(sofa_directory_path)
        self.allowed_keys += ['hrirs']


    @abstractmethod
    def _sofa_path(self, subject_id):
        pass
    

    @abstractmethod
    def _subject_ids(self):
        pass


    def hrir_ids(self, side):
        ids = self._subject_ids()
        if side.startswith('both'):
            if side == 'both-left':
                sides = ('left', 'flipped-right')
            elif side == 'both-right':
                sides = ('flipped-left', 'right')
            else:
                sides = ('left', 'right')
            return [x for x in [(i, s) for i in ids for s in sides]]
        return [x for x in [(i, side) for i in ids]]


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
        selected_azimuths, selected_elevations, row_indices, column_indices = hrir_angle_indices(subject_id, row_angles, column_angles)
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


class AriDataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, anthropomorphy_matfile_path,
                       verbose=False, dtype=np.float32):
        super().__init__(sofa_directory_path, verbose, dtype)
        if anthropomorphy_matfile_path is not None:
            self.anth = io.loadmat(anthropomorphy_matfile_path, squeeze_me=True)
            self.allowed_keys += ['measurements']
        # self.pinna_images_path = Path(pinna_images_path)
        self.dataset = 'ari'


    def _subject_ids(self, exclude=(10,22,826)):
        ids = sorted([int(x.stem.split('_nh')[1]) for x in self.sofa_directory_path.glob('hrtf [bc]_nh*.sofa')])
        try:
            for i in exclude:
                ids.remove(i)
        except ValueError:
            pass
        return ids


    def _sofa_path(self, subject_id):
        return str(next(self.sofa_directory_path.glob('hrtf [bc]_nh{}.sofa'.format(subject_id))))


class ListenDataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, verbose=False, dtype=np.float32):
        super().__init__(sofa_directory_path, verbose, dtype)
        self.dataset = 'listen'


    def _subject_ids(self):
        return sorted([int(x.stem.split('_')[1]) for x in self.sofa_directory_path.glob('IRC_????_C_44100.sofa')])


    def _sofa_path(self, subject_id):
        return str(self.sofa_directory_path / 'IRC_{:04d}_C_44100.sofa'.format(subject_id))


class BiLiDataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, verbose=False, dtype=np.float32):
        super().__init__(sofa_directory_path, verbose, dtype)
        self.dataset = 'bili'


    def _subject_ids(self):
        return sorted([int(x.stem.split('_')[1]) for x in self.sofa_directory_path.glob('IRC_????_C_HRIR_96000.sofa')])


    def _sofa_path(self, subject_id):
        return str(self.sofa_directory_path / 'IRC_{:04d}_C_HRIR_96000.sofa'.format(subject_id))


class ItaDataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, verbose=False, dtype=np.float32):
        super().__init__(sofa_directory_path, verbose, dtype)
        self.dataset = 'ita'


    def _subject_ids(self, exclude=(2,14)):
        ids = sorted([int(x.stem.split('MRT')[1]) for x in self.sofa_directory_path.glob('MRT??.sofa')])
        try:
            for i in exclude:
                ids.remove(i)
        except ValueError:
            pass
        return ids


    def _sofa_path(self, subject_id):
        return str(self.sofa_directory_path / 'MRT{:02d}.sofa'.format(subject_id))


class HutubsDataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, verbose=False, dtype=np.float32):
        super().__init__(sofa_directory_path, verbose, dtype)
        self.dataset = 'hutubs'


    def _subject_ids(self):
        return sorted([int(x.stem.split('_')[0].split('pp')[1]) for x in self.sofa_directory_path.glob('pp??_HRIRs_measured.sofa')])


    def _sofa_path(self, subject_id):
        return str(self.sofa_directory_path / 'pp{:d}_HRIRs_measured.sofa'.format(subject_id))


class RiecDataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, verbose=False, dtype=np.float32):
        super().__init__(sofa_directory_path, verbose, dtype)
        self.dataset = 'riec'


    def _subject_ids(self):
        return sorted([int(Path(x).stem.split('_')[3]) for x in self.sofa_directory_path.glob('RIEC_hrir_subject_???.sofa')])


    def _sofa_path(self, subject_id):
        return str(self.sofa_directory_path / f'RIEC_hrir_subject_{subject_id:03d}.sofa')


class ChedarDataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, radius=1, verbose=False, dtype=np.float32):
        super().__init__(sofa_directory_path, verbose, dtype)
        self.dataset = 'chedar'
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


    def _subject_ids(self):
        return sorted([int(Path(x).stem.split('_')[1]) for x in self.sofa_directory_path.glob(f'chedar_????_UV{self.radius}.sofa')])


    def _sofa_path(self, subject_id):
        return str(self.sofa_directory_path / f'chedar_{subject_id:04d}_UV{self.radius}.sofa')


class WidespreadDataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, radius=1, verbose=False, dtype=np.float32):
        super().__init__(sofa_directory_path, verbose, dtype)
        self.dataset = 'widespread'
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


    def _subject_ids(self):
        return sorted([int(Path(x).stem.split('_')[1]) for x in self.sofa_directory_path.glob(f'UV{self.radius}_?????.sofa')])


    def _sofa_path(self, subject_id):
        return str(self.sofa_directory_path / f'UV{self.radius}_{subject_id:05d}.sofa')


class Sadie2DataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, verbose=False, dtype=np.float32):
        super().__init__(sofa_directory_path, verbose, dtype)
        self.dataset = 'sadie2'


    def _subject_ids(self, exclude=(1,2,3,4,5,6,7,8,9)):
        ids = sorted([int(Path(x).stem[1:]) for x in self.sofa_directory_path.glob('[DH]*')])
        try:
            for i in exclude:
                ids.remove(i)
        except ValueError:
            pass
        return ids


    def _sofa_path(self, subject_id):
        if subject_id < 3:
            sadie2_id = f'D{subject_id}'
        else:
            sadie2_id = f'H{subject_id}'
        return str(self.sofa_directory_path / f'{sadie2_id}/{sadie2_id}_HRIR_SOFA/{sadie2_id}_96K_24bit_512tap_FIR_SOFA.sofa')


class ThreeDThreeADataPoint(SofaDataPoint):

    def __init__(self, sofa_directory_path, verbose=False, dtype=np.float32):
        super().__init__(sofa_directory_path, verbose, dtype)
        self.dataset = '3d3a'


    def _subject_ids(self, exclude=()):
        ids = sorted([int(Path(x).stem.split('_')[0].lstrip('Subject')) for x in self.sofa_directory_path.glob('Subject*_HRIRs.sofa')])
        try:
            for i in exclude:
                ids.remove(i)
        except ValueError:
            pass
        return ids


    def _sofa_path(self, subject_id):
        return str(self.sofa_directory_path / f'Subject{subject_id}_HRIRs.sofa')
