from ..transforms import BatchTransform
from ..util import wrap_closed_open_interval, wrap_closed_interval, lateral_vertical_from_yaw, lateral_vertical_from_pitch, lateral_vertical_from_roll, azimuth_elevation_from_yaw, azimuth_elevation_from_pitch, azimuth_elevation_from_roll, nextpow2, vectormap
from abc import ABC, abstractmethod
from typing import Iterable, Optional
import numpy as np
import numpy.typing as npt
from scipy.fft import rfft, fft, ifft
from scipy.signal import get_window, hilbert
import soxr


def _to_dense2d(multidim_array: np.ndarray) -> np.ndarray:
    ''' Converts an N-D np.ndarray to a 2-D dense array which flattens all dimensions except the last into one while keeping the last 
        dimension intact.'''
    dense_array = multidim_array.compressed() if np.ma.isMaskedArray(multidim_array) else multidim_array
    return dense_array.reshape(-1, multidim_array.shape[-1])


def _to_dense3d(multidim_array: np.ndarray) -> np.ndarray:
    ''' Converts an N-D np.ndarray to a 3-D dense array which flattens the middle dimensions while keeping the first and last 
        dimension intact.'''
    dense_array = multidim_array.compressed() if np.ma.isMaskedArray(multidim_array) else multidim_array
    return dense_array.reshape(len(multidim_array), -1, multidim_array.shape[-1])


def _to_multidim(dense2d_array: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    ''' Converts a 2-D dense np.ndarray into a N-D np.ndarray whose dimensions and sparsity mask is taken from the given prototypical 
        sparse array, except for the final dimension which is taken from the dense array itself.'''
    if np.ma.getmaskarray(prototype).any():
        sparse_mask = np.tile(np.ma.getmaskarray(prototype)[..., :1], (*np.ones(prototype.ndim-1, dtype=int), dense2d_array.shape[-1]))
        sparse_array = np.ma.array(np.empty((*prototype.shape[:-1], dense2d_array.shape[-1])), dtype=dense2d_array.dtype, mask=sparse_mask)
        sparse_array[~sparse_mask] = dense2d_array.ravel()
        return sparse_array
    else:
        return dense2d_array.reshape(*prototype.shape[:-1], -1)


class FlattenPositionsTransform(BatchTransform):
    def __init__(self, mask: Optional[np.ndarray] = None):
        self._mask = mask


    def __call__(self, hrirs: np.ndarray) -> np.ndarray:
        if self._mask is None:
            self._mask = np.ma.getmaskarray(hrirs)[..., :1]
        dense_hrirs = hrirs.compressed() if np.ma.isMaskedArray(hrirs) else hrirs
        return dense_hrirs.reshape(*hrirs.shape[:-4], -1, hrirs.shape[-1])


class Hrir2dTransform(BatchTransform):
    def __init__(self, prototype: Optional[np.ndarray] = None):
        self._prototype = prototype


    def __call__(self, hrirs: np.ndarray) -> np.ndarray:
        if self._prototype is None:
            self._prototype = hrirs
        return _to_dense2d(hrirs)


class Hrir3dTransform(BatchTransform):
    def __init__(self, prototype: Optional[np.ndarray] = None):
        self._prototype = prototype


    def __call__(self, hrirs: np.ndarray) -> np.ndarray:
        if self._prototype is None:
            self._prototype = hrirs
        return _to_dense3d(hrirs)


class ScaleTransform(BatchTransform):
    def __init__(self, additive_factor: float = 0, multiplicative_factor = 1):
        self.additive_factor = additive_factor
        self.multiplicative_factor = multiplicative_factor


    def __call__(self, values: np.ndarray) -> np.ndarray:
        scaled_values = values
        if isinstance(self.additive_factor, np.ndarray) or self.additive_factor != 0:
            scaled_values = scaled_values + self.additive_factor
        if isinstance(self.multiplicative_factor, np.ndarray) or self.multiplicative_factor != 1:
            scaled_values = scaled_values * self.multiplicative_factor
        return scaled_values


class MinPhaseTransform(BatchTransform):
    def __call__(self, hrirs: np.ndarray) -> np.ndarray:
        dense_hrirs = _to_dense2d(hrirs)
        hrtf = fft(dense_hrirs)
        magnitudes = np.abs(hrtf)
        min_phases = -np.imag(hilbert(np.log(np.maximum(magnitudes, np.finfo(magnitudes.dtype).smallest_subnormal))))
        min_phase_hrtf = magnitudes * np.exp(1j * min_phases)
        min_phase_hrirs = np.real(ifft(min_phase_hrtf)[..., :hrirs.shape[-1]])
        return _to_multidim(min_phase_hrirs, hrirs)


class ResampleTransform(BatchTransform):
    def __init__(self, original_rate:float, target_rate: float):
        self.original_rate = original_rate
        self.target_rate = target_rate


    def __call__(self, hrirs: np.ndarray) -> np.ndarray:
        if self.original_rate == self.target_rate:
            return hrirs
        dense_hrirs = _to_dense2d(hrirs)
        # process in chunks of 65536 HRIRs (channels) because that's the maximum supported by soxr
        resampled_hrirs = np.vstack([soxr.resample(dense_hrirs[ch_idx:ch_idx+soxr._CH_LIMIT].T, self.original_rate, self.target_rate).T for ch_idx in range(0, len(dense_hrirs), soxr._CH_LIMIT)])
        return _to_multidim(resampled_hrirs, hrirs)


class SelectIndicesTransform(BatchTransform):

    def __init__(self, indices):
        self._selection = indices


    def __call__(self, values: np.ndarray) -> np.ndarray:
        return values[..., self._selection]


class TruncateTransform(SelectIndicesTransform):
    def __init__(self, truncated_length: int | None = None, fadeout_length: int = 0, fadein_length: int = 0, start: int = 0, window_type: str = 'hann'):
        super().__init__(slice(start, None if truncated_length is None else start + truncated_length))
        self._truncated_length = truncated_length
        self._fadeout_length = fadeout_length
        self._fadein_length = fadein_length
        self._window_type = window_type


    def __call__(self, values: np.ndarray) -> np.ndarray:
        truncated_values = super().__call__(values)
        if self._fadein_length > 0:
            fadein_window = get_window(self._window_type, self._fadein_length * 2 + 1, False)[:self._fadein_length]
            truncated_values[..., :self._fadein_length] *= fadein_window
        if self._fadeout_length > 0:
            fadeout_window = get_window(self._window_type, self._fadeout_length * 2 + 1, False)[-self._fadeout_length:]
            truncated_values[..., -self._fadeout_length:] *= fadeout_window
        return truncated_values


class PadTransform(BatchTransform):
    def __init__(self, append_samples: int = 0, prepend_samples: int = 0, mode: str = 'constant', **pad_kwargs):
        self._pad_width = (prepend_samples, append_samples)
        self._mode = mode
        self._pad_kwargs = pad_kwargs

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return np.pad(values, {-1: self._pad_width}, self._mode, **self._pad_kwargs)


class DecibelTransform(BatchTransform):
    def __call__(self, hrtf: np.ndarray) -> np.ndarray:
        # limit dB range to what is representable by data type
        min_value = np.max(hrtf) * np.finfo(hrtf.dtype).resolution
        return 20 * np.log10(np.maximum(hrtf, min_value))


class DomainTransform(BatchTransform):
    def __init__(self, domain: str, dtype: Optional[npt.DTypeLike]=None):
        if domain not in ('time', 'complex', 'magnitude', 'power_db', 'phase'):
            raise ValueError(f'Unknown domain "{domain}" for HRIRs')
        if domain == 'complex' and not issubclass(dtype, np.complexfloating):
            raise ValueError(f'An HRTF in the complex domain requires the dtype to be set to a complex type (currently {dtype})')
        if domain.endswith('_db'):
            self._db_transformer = DecibelTransform()
        self.domain = domain
        self.dtype = dtype


    def __call__(self, hrirs: np.ndarray) -> np.ndarray:
        if self.domain == 'time':
            transformed_hrirs = hrirs
        else:
            dense_hrirs = _to_dense2d(hrirs)
            dense_hrtf = rfft(dense_hrirs, norm='forward')
            hrtf = _to_multidim(dense_hrtf, hrirs)
            if self.domain == 'complex':
                transformed_hrirs = hrtf
            elif self.domain in ('magnitude', 'power_db'):
                if np.ma.isMaskedArray(hrtf):
                    # workaround to avoid spurious warning triggered by np.abs in NumPy v1.23.5
                    hrtf._fill_value = hrirs.fill_value
                magnitudes = np.abs(hrtf)
                if self.domain.endswith('_db'):
                    transformed_hrirs = self._db_transformer(magnitudes)
                else:
                    transformed_hrirs = magnitudes
            elif self.domain == 'phase':
                transformed_hrirs = np.angle(hrtf)
        if self.dtype is None:
            return transformed_hrirs
        return transformed_hrirs.astype(self.dtype)


class SelectValueRangeTransform(SelectIndicesTransform):

    def __init__(self, value_legend, lower_limit, upper_limit):
        super().__init__((value_legend >= lower_limit) & (value_legend < upper_limit))


class PlaneTransform(ABC):

    def __init__(self, plane, plane_offset, positive_angles, fundamental_angles, orthogonal_angles, selection_mask):
        self.plane = plane
        self.plane_offset = plane_offset
        self.fundamental_angles = fundamental_angles
        self.orthogonal_angles = orthogonal_angles
        self.selection_mask = selection_mask
        self.positive_angles = positive_angles # triggers calculation of min_angle, max_angle & plane_angles
        if plane in ('horizontal', 'interaural'):
            self.plane_angle_name = 'yaw [°]'
        elif plane in ('median', 'vertical'):
            self.plane_angle_name = 'pitch [°]'
        else: # frontal plane
            self.plane_angle_name = 'roll [°]'


    @property
    def positive_angles(self):
        return self._positive_angles


    @positive_angles.setter
    def positive_angles(self, value: bool):
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
        self.plane_angles = self._calc_plane_angles()


    @staticmethod
    @abstractmethod
    def convert_plane_angles(
        plane: str,
        plane_angles: Optional[Iterable[float]],
        plane_offset: float,
    ):
        pass


    @abstractmethod
    def process(self, hrirs: np.ndarray):
        pass


    def __call__(self, single_plane: np.ma.MaskedArray):
        single_plane = self.process(single_plane)
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
    def _calc_plane_angles(self):
        pass
    

    def _stitch_plane(self, selected_angles):
        plane_angles = self(selected_angles)
        return wrap_closed_open_interval(plane_angles, self.min_angle, self.max_angle)


    def __repr__(self):
        return self.__class__.__name__ + '()'


    @staticmethod
    def _idx_first_not_smaller_than(iterable, value=0):
        try:
            return (iterable >= value).argmax()
        except IndexError:
            return len(iterable)


    @staticmethod
    def _idx_first_larger_than(iterable, value=0):
        try:
            return (iterable > value).argmax()
        except IndexError:
            return len(iterable)


class InterauralPlaneTransform(PlaneTransform):

    _split_idx: int
    _left_pole_overlap: bool
    _right_pole_overlap: bool
    _pos_sphere_present: bool
    _neg_sphere_present: bool


    @staticmethod
    def convert_plane_angles(
        plane: str,
        plane_angles: Optional[Iterable[float]],
        plane_offset: float,
    ):
        if plane == 'horizontal' or plane == 'interaural':
            if plane == 'horizontal' and plane_offset != 0:
                raise ValueError('Only the horizontal plane at vertical angle 0 is available in an interaural grid dataset')
            if plane_angles is None:
                lateral_angles, vertical_angles = None, (plane_offset - 180, plane_offset)
            else:
                lateral_angles, vertical_angles = lateral_vertical_from_yaw(plane_angles, plane_offset)
        elif plane == 'median':
            if plane_angles is None:
                lateral_angles, vertical_angles = (plane_offset,), None
            else:
                lateral_angles, vertical_angles = lateral_vertical_from_pitch(plane_angles, plane_offset)
        elif plane == 'frontal':
            if plane_offset != 0:
                raise ValueError('Only the frontal plane at vertical angles +/-90 is available in an interaural grid dataset')
            if plane_angles is None:
                lateral_angles, vertical_angles = None, (plane_offset - 90, plane_offset + 90)
            else:
                lateral_angles, vertical_angles = lateral_vertical_from_roll(plane_angles, plane_offset)
        else:
            if plane not in ('horizontal', 'median', 'frontal', 'interaural'):
                raise ValueError('Unknown plane "{}", needs to be "horizontal", "median", "frontal" or "interaural".')
        return vertical_angles, lateral_angles


    def _calc_plane_angles(self):
        if self.selection_mask is None:
            return np.array([])
        vertical_angles = self.fundamental_angles
        lateral_angles = np.ma.array(self.orthogonal_angles, mask=~self.selection_mask[0])

        if self.plane == 'median':
            input_angles = vertical_angles.tolist()
            if self.positive_angles:
                self._split_idx = self._idx_first_not_smaller_than(vertical_angles)
            else:
                self._split_idx = self._idx_first_not_smaller_than(vertical_angles, -90)
        else:
            self._split_idx = self._idx_first_not_smaller_than(self.orthogonal_angles)
            if len(vertical_angles) > 1:
                # both half planes present
                back_yaw_angles = 180-lateral_angles
                front_yaw_angles = np.ma.array(self.orthogonal_angles, mask=~self.selection_mask[1])
                input_angles = [back_yaw_angles, front_yaw_angles]
                self._left_pole_overlap = np.isclose(front_yaw_angles, 90).any() and np.isclose(back_yaw_angles, 90).any()
                self._right_pole_overlap = np.isclose(front_yaw_angles, -90).any() and np.isclose(back_yaw_angles, 270).any()
                self._pos_sphere_present = True
                self._neg_sphere_present = True
            else:
                # floating point safe version of check below
                # if vertical_angles <= 90 and vertical_angles > -90:
                rtol = 1e-5
                atol = 1e-8
                upper_lim = vertical_angles[0] - 90
                lower_lim = vertical_angles[0] + 90
                if (upper_lim <= rtol*np.abs(upper_lim)+atol and lower_lim > rtol*np.abs(lower_lim)+atol).item():
                    # only front half plane is present
                    self._pos_sphere_present = True
                    self._neg_sphere_present = False
                    input_angles = [lateral_angles]
                else:
                    # only back half plane is present
                    self._pos_sphere_present = False
                    self._neg_sphere_present = True
                    input_angles = [180 - lateral_angles]
                self._left_pole_overlap = False
                self._right_pole_overlap = False
            if self.plane == 'frontal':
                # reverse angular direction
                input_angles = [-x for x in input_angles]

        return super()._stitch_plane(input_angles)


    def process(self, hrirs: np.ndarray):
        if self.plane == 'median':
            if self.positive_angles:
                down, up = np.split(hrirs, [self._split_idx])
                single_plane = np.ma.concatenate((up, down))
            else:
                back_down, rest = np.split(hrirs, [self._split_idx])
                single_plane = np.ma.concatenate((rest, back_down))
            if single_plane.ndim > 1:
                single_plane = np.squeeze(single_plane, axis=(1, 2))
        else:
            if self._pos_sphere_present and self._neg_sphere_present:
                half1, half2 = hrirs
                if self._left_pole_overlap:
                    left_pole_mask = np.full(half2.shape, False)
                    left_pole_mask[-1] = True
                    half2 = np.ma.MaskedArray(half2, left_pole_mask)
                if self._right_pole_overlap:
                    right_pole_mask = np.full(half1.shape, False)
                    right_pole_mask[0] = True
                    half1 = np.ma.MaskedArray(half1, right_pole_mask)
            if self.plane == 'frontal':
                if self._pos_sphere_present and self._neg_sphere_present:
                    # both half planes present
                    down_right_left = half1
                    up_left_right = np.flip(half2, axis=0)
                    if self.positive_angles:
                        left_up, right_up = np.split(up_left_right, [-self._split_idx-1])
                        single_plane = np.ma.concatenate((right_up, down_right_left, left_up))
                    else:
                        right_down, left_down = np.split(down_right_left, [self._split_idx])
                        single_plane = np.ma.concatenate((left_down, up_left_right, right_down))
                elif self._pos_sphere_present:
                    # only up half plane present
                    up_left_right = np.flip(hrirs[0], axis=0)
                    if self.positive_angles:
                        left_up, right_up = np.split(up_left_right, [-self._split_idx-1])
                        single_plane = np.ma.concatenate((right_up, left_up))
                    else:
                        single_plane = up_left_right
                else:
                    # only down half plane present
                    down_right_left = hrirs[0]
                    if self.positive_angles:
                        single_plane = down_right_left
                    else:
                        right_down, left_down = np.split(down_right_left, [self._split_idx])
                        single_plane = np.ma.concatenate((left_down, right_down))
            else:
                if self._pos_sphere_present and self._neg_sphere_present:
                    # both half planes present
                    back_left_right = np.flip(half1, axis=0)
                    front_right_left = half2
                    if self.positive_angles:
                        front_right, front_left = np.split(front_right_left, [self._split_idx])
                        single_plane = np.ma.concatenate((front_left, back_left_right, front_right))
                    else:
                        back_left, back_right = np.split(back_left_right, [-self._split_idx-1])
                        single_plane = np.ma.concatenate((back_right, front_right_left, back_left))
                elif self._pos_sphere_present:
                    # only front half plane present
                    front_right_left = hrirs[0]
                    if self.positive_angles:
                        front_right, front_left = np.split(front_right_left, [self._split_idx])
                        single_plane = np.ma.concatenate((front_left, front_right))
                    else:
                        single_plane = front_right_left
                else:
                    # only back half plane present
                    back_left_right = np.flip(hrirs[0], axis=0)
                    if self.positive_angles:
                        single_plane = back_left_right
                    else:
                        back_left, back_right = np.split(back_left_right, [-self._split_idx-1])
                        single_plane = np.ma.concatenate((back_right, back_left))
            if single_plane.ndim > 1:
                single_plane = np.squeeze(single_plane, axis=1)

        return single_plane


class SphericalPlaneTransform(PlaneTransform):

    _split_idx: int
    _up_pole_overlap: bool
    _down_pole_overlap: bool
    _pos_sphere_present: bool
    _neg_sphere_present: bool


    @staticmethod
    def convert_plane_angles(
        plane: str,
        plane_angles: Optional[Iterable[float]],
        plane_offset: float,
    ):
        if plane == 'horizontal':
            if plane_angles is None:
                azimuth_angles, elevation_angles = None, (plane_offset,)
            else:
                azimuth_angles, elevation_angles = azimuth_elevation_from_yaw(plane_angles, plane_offset)
        elif plane == 'median' or plane == 'vertical':
            if plane == 'median' and plane_offset != 0:
                raise ValueError('Only the median plane at azimuth 0 is available in a spherical coordinate dataset')
            if plane_angles is None:
                azimuth_angles, elevation_angles = (plane_offset - 180, plane_offset), None
            else:
                azimuth_angles, elevation_angles = azimuth_elevation_from_pitch(plane_angles, plane_offset)
        elif plane == 'frontal':
            if plane_offset != 0:
                raise ValueError('Only the frontal plane at azimuth +/-90 is available in a spherical coordinate dataset')
            if plane_angles is None:
                azimuth_angles, elevation_angles = (plane_offset - 90, plane_offset + 90), None
            else:
                azimuth_angles, elevation_angles = azimuth_elevation_from_roll(plane_angles, plane_offset)
        else:
            raise ValueError('Unknown plane "{}", needs to be "horizontal", "median", "frontal" or "vertical".')
        return azimuth_angles, elevation_angles


    def _calc_plane_angles(self):
        if self.selection_mask is None:
            return np.array([])
        azimuth_angles = self.fundamental_angles
        elevation_angles = np.ma.array(self.orthogonal_angles, mask=~self.selection_mask[0])

        if self.plane == 'horizontal':
            input_angles = azimuth_angles.tolist()
            self._split_idx = self._idx_first_not_smaller_than(azimuth_angles)
        else:
            self._split_idx = self._idx_first_not_smaller_than(self.orthogonal_angles)
            if len(azimuth_angles) > 1:
                # both half planes present
                back_pitch_angles = 180-elevation_angles
                front_pitch_angles = np.ma.array(self.orthogonal_angles, mask=~self.selection_mask[1])
                input_angles = [back_pitch_angles, front_pitch_angles]
                self._up_pole_overlap = np.isclose(front_pitch_angles, 90).any() and np.isclose(back_pitch_angles, 90).any()
                self._down_pole_overlap = np.isclose(front_pitch_angles, -90).any() and np.isclose(back_pitch_angles, 270).any()
                self._pos_sphere_present = True
                self._neg_sphere_present = True
            else:
                # floating point safe version of check below
                # if azimuth_angles <= 90 and azimuth_angles > -90:
                rtol = 1e-5
                atol = 1e-8
                upper_lim = azimuth_angles[0] - 90
                lower_lim = azimuth_angles[0] + 90
                if (upper_lim <= rtol*np.abs(upper_lim)+atol and lower_lim > rtol*np.abs(lower_lim)+atol).item():
                    # only front half plane is present
                    self._pos_sphere_present = True
                    self._neg_sphere_present = False
                    input_angles = [elevation_angles]
                else:
                    # only back half plane is present
                    self._pos_sphere_present = False
                    self._neg_sphere_present = True
                    input_angles = [180 - elevation_angles]
                self._up_pole_overlap = False
                self._down_pole_overlap = False
            if self.plane == 'frontal':
                # shift origin from Y to Z axis
                input_angles = [x - 90 for x in input_angles]
        
        return super()._stitch_plane(input_angles)


    def process(self, hrirs: np.ndarray):
        if self.plane == 'horizontal':
            if self.positive_angles:
                right, left = np.split(hrirs, [self._split_idx])
                single_plane = np.ma.concatenate((left, right))
            else:
                single_plane = np.asanyarray(hrirs)
            if single_plane.ndim > 1:
                single_plane = np.squeeze(single_plane, axis=(1, 2))
        else:
            if self._pos_sphere_present and self._neg_sphere_present:
                half1, half2 = hrirs
                if self._up_pole_overlap:
                    up_pole_mask = np.full(half2.shape, False)
                    up_pole_mask[-1] = True
                    half2 = np.ma.MaskedArray(half2, up_pole_mask)
                if self._down_pole_overlap:
                    down_pole_mask = np.full(half1.shape, False)
                    down_pole_mask[0] = True
                    half1 = np.ma.MaskedArray(half1, down_pole_mask)
            if self.plane == 'frontal':
                if self._pos_sphere_present and self._neg_sphere_present:
                    # both right and left half planes present
                    right_up_down = np.flip(half1, axis=0)
                    left_down_up = half2
                    if self.positive_angles:
                        single_plane = np.ma.concatenate((right_up_down, left_down_up))
                    else:
                        single_plane = np.ma.concatenate((left_down_up, right_up_down))
                elif self._pos_sphere_present:
                    # only left half plane present
                    left_down_up = hrirs[0]
                    single_plane = left_down_up
                else:
                    # only right half plane present
                    right_up_down = np.flip(hrirs[0], axis=0)
                    single_plane = right_up_down
            else:
                if self._pos_sphere_present and self._neg_sphere_present:
                    # both back and front half planes present
                    back_up_down = np.flip(half1, axis=0)
                    front_down_up = half2
                    if not self.positive_angles:
                        single_plane = np.ma.concatenate((front_down_up, back_up_down))
                    else:
                        front_down, front_up = np.split(front_down_up, [self._split_idx])
                        single_plane = np.ma.concatenate((front_up, back_up_down, front_down))
                elif self._pos_sphere_present:
                    # only front half plane present
                    front_down_up = hrirs[0]
                    if self.positive_angles:
                        front_down, front_up = np.split(front_down_up, [self._split_idx])
                        single_plane = np.ma.concatenate((front_up, front_down))
                    else:
                        single_plane = front_down_up
                else:
                    # only back half plane present
                    back_up_down = np.flip(hrirs[0], axis=0)
                    single_plane = back_up_down
            if single_plane.ndim > 1:
                single_plane = np.squeeze(single_plane, axis=1)

        return single_plane


class CompensateFreefieldTransform(BatchTransform):
    def __init__(self, freefield_median_plane: np.ndarray, freefield_elevations: np.ndarray, ds_azimuths: np.ndarray, ds_elevations: np.ndarray, delay_samples: int,
                 *, truncated_samples: int | None = None, fadein_samples: int = 0, fadeout_samples: int = 0, append_samples: int = 0):
        self._freefield_hrirs = self.assemble_freefield_tensor(freefield_median_plane, freefield_elevations, ds_azimuths, ds_elevations)
        self._delay = delay_samples
        self._truncate = TruncateTransform(truncated_length=truncated_samples, fadein_length=fadein_samples, fadeout_length=fadeout_samples)
        self._pad = PadTransform(append_samples=append_samples)
        self.freefield_length = self._freefield_hrirs.shape[-1]
        self.truncated_samples = truncated_samples
        self.append_samples = append_samples


    def __call__(self, hrirs: np.ndarray) -> np.ndarray:
        num_tiles = len(hrirs) // len(self._freefield_hrirs)
        if len(self._freefield_hrirs) * num_tiles != len(hrirs):
            raise ValueError(f'The number of measured ears ({len(hrirs)})is not a multiple of the number of freefield ears ({len(self._freefield_hrirs)}) ')
        tiled_freefield_hrirs = np.tile(self._freefield_hrirs, (num_tiles,)+(1,)*(self._freefield_hrirs.ndim-1))

        conv_len = hrirs.shape[-1] + self._freefield_hrirs.shape[-1] - 1
        fft_len = nextpow2(conv_len)
        circ_conv = np.fft.irfft(np.fft.rfft(hrirs, n=fft_len) * np.fft.rfft(tiled_freefield_hrirs, n=fft_len))
        shifted_conv = np.roll(circ_conv, shift=(-self._delay), axis=-1)[..., :conv_len]
        sparse_mask = np.tile(np.ma.getmaskarray(hrirs)[..., :1], conv_len)
        masked_conv = np.ma.array(shifted_conv, dtype=hrirs.dtype, mask=sparse_mask)

        truncated_hrirs = self._truncate(masked_conv)
        padded_hrirs = self._pad(truncated_hrirs)
        return padded_hrirs


    @staticmethod
    def assemble_freefield_tensor(freefield_median_plane, freefield_elevations, ds_azimuths, ds_elevations):
        freefield_elevations = wrap_closed_open_interval(freefield_elevations, -90, 270)
        ds_azimuths = wrap_closed_open_interval(ds_azimuths, -180, 180)
        ds_elevations = wrap_closed_interval(ds_elevations, -90, 90)
        if freefield_median_plane.ndim == 2:
            freefield_median_plane = np.expand_dims(freefield_median_plane, 0)
        front_elevation_map = vectormap(ds_elevations, freefield_elevations)
        front_halfplane = np.expand_dims(freefield_median_plane[:, front_elevation_map], axis=1)
        if any(freefield_elevations > 90):
            # Front and back freefield measurements
            back_elevation_map = vectormap(180-ds_elevations, freefield_elevations)
            back_halfplane = np.expand_dims(freefield_median_plane[:, back_elevation_map], axis=1)
            halfplanes = []
            if -180 in ds_azimuths:
                halfplanes.append(back_halfplane)
            halfplanes.extend([front_halfplane] * sum((ds_azimuths > -180) & (ds_azimuths <= 0)))
            halfplanes.extend([back_halfplane] * sum(ds_azimuths > 0))
            freefield_hrirs = np.concatenate(halfplanes, axis=1)
        else:
            # Front freefield measurement only
            freefield_hrirs = np.tile(front_halfplane, (len(ds_azimuths)//2, 1, 1))
        return np.expand_dims(freefield_hrirs, axis=3) # add distance dimension
