from .datareader import DataReader, CipicDataReader, AriDataReader, ListenDataReader, BiLiDataReader, CrossModDataReader, ItaDataReader, HutubsDataReader, RiecDataReader, ChedarDataReader, WidespreadDataReader, Sadie2DataReader, Princeton3D3ADataReader, ScutDataReader, SonicomDataReader, MitKemarDataReader, CustomSphericalDataReader
from .display import plot_hrir_plane, plot_hrtf_plane, plot_plane_positions, plot_3d_positions, plot_hrir_lines, plot_hrtf_lines
from .specifications import Spec, HrirSpec, sanitise_specs, sanitise_multiple_specs
from .transforms.hrir import BatchTransform, ScaleTransform, MinPhaseTransform, ResampleTransform, TruncateTransform, DomainTransform, SelectValueRangeTransform, PlaneTransform, GriddedPositionsTransform, PadTransform, CompensateFreefieldTransform
from .util import quantise
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from itertools import chain
from pathlib import Path
from numbers import Integral, Real
from typing import Dict, Iterable, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt
from scipy.fft import rfftfreq
from tqdm.auto import tqdm


def _insert(transforms: list[BatchTransform], orig_len: int, transform: BatchTransform):
    return transforms.insert(len(transforms) - orig_len, transform)


class Dataset:

    def __init__(
        self,
        datareader: DataReader,
        features_spec: Union[Spec, Iterable[Spec]],
        target_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        group_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: npt.DTypeLike = np.float32,
        dict_format = True,
    ) -> None:
        super().__init__()
        self.datareader = datareader
        self.fundamental_angle_name = datareader.fundamental_angle_name
        self.orthogonal_angle_name = datareader.orthogonal_angle_name
        self._coordinate_transform = datareader._coordinate_transform
        # Allow specifying ids that are excluded by default without explicitly overriding `exclude_ids``
        if subject_ids is not None and not isinstance(subject_ids, str) and exclude_ids is None:
            exclude_ids = ()
        self._exclude_ids = exclude_ids
        self._dict_format = dict_format
        self.dtype = dtype

        features_spec = sanitise_specs(features_spec)
        target_spec = sanitise_specs(target_spec)
        group_spec = sanitise_specs(group_spec)

        self._specification = {spec.name: spec.spec for spec in chain(features_spec, target_spec, group_spec)}
        if not self._specification:
            raise ValueError('At least one specification should be given.')
        self._features_keys = tuple((spec.name for spec in features_spec))
        self._target_keys = tuple((spec.name for spec in target_spec))
        self._group_keys = tuple((spec.name for spec in group_spec))

        if subject_requirements is not None:
            self._specification = {**self._specification, **subject_requirements}
        self.collection_id = datareader.query.collection_id(self._specification.get('hrir', {}).get('method'), self._specification.get('hrir', {}).get('variant'))
        ear_ids = datareader.query.specification_based_ids(self._specification, include_subjects=subject_ids, exclude_subjects=exclude_ids)

        if len(ear_ids) == 0:
            if len(datareader.query.specification_based_ids(self._specification)) == 0:
                raise ValueError('Empty dataset. Check if its configuration and paths are correct.')
            if subject_ids:
                raise ValueError('None of the explicitly requested subject IDs are available.')
            self.subject_ids: Tuple[Union[int, str], ...] = tuple()
            self.hrir_samplerate = None
            self._hrir_length = None
            self.fundamental_angles = np.array([])
            self.orthogonal_angles = np.array([])
            self.radii = np.array([])
            self.grid_mask = np.array([])
            self._data = {}
            return

        self.subject_ids, self.sides = zip(*ear_ids)

        if 'hrir' in self._specification.keys():
            hrir_spec = self._specification['hrir']
            # Handle planar specification
            if 'plane' in hrir_spec:
                requested_fundamental_angles, requested_orthogonal_angles = datareader.PlaneTransform.convert_plane_angles(hrir_spec['plane'], hrir_spec.get('plane_angles'), hrir_spec['plane_offset'])
            else:
                requested_fundamental_angles, requested_orthogonal_angles = hrir_spec.get('fundamental_angles'), hrir_spec.get('orthogonal_angles')
            # Verify angle symmetry before reading complete dataset
            self.fundamental_angles, self.orthogonal_angles, self.radii, self.grid_mask, *_ = datareader._map_sofa_position_order_to_matrix(
                self.subject_ids[0], requested_fundamental_angles, requested_orthogonal_angles, hrir_spec.get('distance'),
            )
            if hrir_spec.get('side', '').startswith('any-'):
                datareader._verify_angle_symmetry(self.fundamental_angles, self.orthogonal_angles)
            # Create plane transform from file angles and mask
            if 'plane' in hrir_spec:
                self._plane_transform = datareader.PlaneTransform(hrir_spec['plane'], hrir_spec['plane_offset'], hrir_spec['positive_angles'], self.fundamental_angles, self.orthogonal_angles, self.grid_mask)
                self.append_transform(HrirSpec, self._plane_transform)
            elif not hrir_spec['flat_positions']:
                self.append_transform(HrirSpec, GriddedPositionsTransform(self.grid_mask))
            # Construct HRIR processing pipeline
            num_preprocess = len(hrir_spec['preprocess'])
            recorded_hrir_length = datareader.hrir_length(self.subject_ids[0])
            recorded_samplerate = datareader.hrir_samplerate(self.subject_ids[0])
            if hrir_spec.get('additive_scale_factor') is not None or hrir_spec.get('multiplicative_scale_factor') is not None:
                _insert(hrir_spec['preprocess'], num_preprocess, ScaleTransform(hrir_spec.get('additive_scale_factor', 0), hrir_spec.get('multiplicative_scale_factor', 1)))
            if hrir_spec['min_phase']:
                _insert(hrir_spec['preprocess'], num_preprocess, MinPhaseTransform())
            if hrir_spec.get('samplerate') is None:
                self.hrir_samplerate = recorded_samplerate
            else:
                self.hrir_samplerate = hrir_spec['samplerate']
                _insert(hrir_spec['preprocess'], num_preprocess, ResampleTransform(recorded_samplerate, self.hrir_samplerate))
            if hrir_spec.get('length') is not None:
                if isinstance(hrir_spec['length'], dict):
                    truncate_keys = {'truncated_length', 'fadeout_length', 'fadein_length', 'start', 'window_type'}
                    truncate_kwargs, pad_kwargs = {}, {}
                    for k, v in hrir_spec['length'].items():
                        (truncate_kwargs if k in truncate_keys else pad_kwargs)[k] = v
                    if truncate_kwargs:
                        _insert(hrir_spec['preprocess'], num_preprocess, TruncateTransform(**truncate_kwargs))
                    if pad_kwargs:
                        _insert(hrir_spec['preprocess'], num_preprocess, PadTransform(**pad_kwargs))
                else:
                    _insert(hrir_spec['preprocess'], num_preprocess, TruncateTransform(hrir_spec['length']))
            self._hrir_length = round(recorded_hrir_length * self.hrir_samplerate / recorded_samplerate)
            if self.hrir_length > self._hrir_length:
                raise ValueError(f'Requested HRIR length ({self.hrir_length}) exceeds available length ({self._hrir_length}) at sample rate {self.hrir_samplerate} Hz.')
            self._hrtf_frequencies = rfftfreq(self.hrir_length, 1/self.hrir_samplerate)
            if (hrir_spec['domain'] is not None and hrir_spec['domain'] != 'time') or self.dtype is not None:
                _insert(hrir_spec['preprocess'], num_preprocess, DomainTransform(hrir_spec['domain'], self.dtype))
            if (hrir_spec['domain'] is not None or hrir_spec['domain'] != 'time') and hrir_spec.get('min_freq') is not None or hrir_spec.get('max_freq') is not None:
                min_freq = hrir_spec['min_freq'] if hrir_spec.get('min_freq') is not None else 0
                max_freq = hrir_spec['max_freq'] if hrir_spec.get('max_freq') is not None else np.inf
                _insert(hrir_spec['preprocess'], num_preprocess, SelectValueRangeTransform(self._hrtf_frequencies, min_freq, max_freq))
        self.verify_chain()

        self._data = defaultdict(list)
        for subject, side in tqdm(ear_ids, desc='Loading data from disk', leave=None):
            if 'image' in self._specification.keys():
                self._data['image'].append(datareader.image(subject, side=side, rear=self._specification['image']['rear']))
            if 'anthropometry' in self._specification.keys():
                self._data['anthropometry'].append(datareader.anthropometric_data(subject, side=side, select=self._specification['anthropometry'].get('select')).astype(self.dtype))
            if 'hrir' in self._specification.keys():
                self._data['hrir'].append(datareader.hrir(subject, side, requested_fundamental_angles, requested_orthogonal_angles, self._specification['hrir'].get('distance')))
            if 'subject' in self._specification.keys():
                self._data['subject'].append(subject)
            if 'side' in self._specification.keys():
                self._data['side'].append(side)
            if 'collection' in self._specification.keys():
                self._data['collection'].append(self.collection_id)

        for k in self._specification.keys():
            if k in ('hrir', 'anthropometry'):
                try:
                    self._data[k] = np.ma.stack(self._data[k])
                except ValueError:
                    raise ValueError(f'Not all data points have the same {k} shape') from None
            for preprocess in tqdm(self._specification[k]['preprocess'], desc=f'Preprocessing {k}', leave=None):
                if isinstance(preprocess, BatchTransform):
                    self._data[k] = preprocess(self._data[k])
                else:
                    self._data[k] = np.ma.stack(tuple(map(preprocess, self._data[k])))


    def __len__(self):
        try:
            return len(tuple(self._data.values())[0])
        except IndexError:
            return 0


    def _shape_data(self, keys, data):
        if len(keys) == 0:
            return np.array([], dtype=self.dtype)
        if len(keys) == 1:
            return data[list(keys)[0]]
        return tuple(data[k] for k in keys)


    def _transform_data(self, keys, idx):
        transformed_data = {}
        for k in keys:
            try:
                transformed_data[k] = self._data[k][idx]
            except IndexError:
                raise IndexError('Dataset index out of range') from None
            for transform in self._specification[k]['transform']:
                if isinstance(idx, Integral) or isinstance(transform, BatchTransform):
                    transformed_data[k] = transform(transformed_data[k])
                else:
                    transformed_data[k] = np.ma.stack(tuple(map(transform, transformed_data[k])))
        return transformed_data


    @property
    def features(self):
        transformed_data = self._transform_data(self._features_keys, slice(None))
        return self._shape_data(self._features_keys, transformed_data)


    @property
    def target(self):
        transformed_data = self._transform_data(self._target_keys, slice(None))
        return self._shape_data(self._target_keys, transformed_data)


    @property
    def group(self):
        transformed_data = self._transform_data(self._group_keys, slice(None))
        return self._shape_data(self._group_keys, transformed_data)


    def __getitem__(self, idx):
        # Apply dynamic transform to selected data
        transformed_data = self._transform_data(self._specification.keys(), idx)

        # Assemble features, targets and groups from transformed data
        features = self._shape_data(self._features_keys, transformed_data)
        target = self._shape_data(self._target_keys, transformed_data)
        group = self._shape_data(self._group_keys, transformed_data)

        if self._dict_format:
            return {'features': features, 'target': target, 'group': group}
        elif group.size == 0:
            return (features, target)
        else:
            return (features, target, group)


    @property
    def available_subject_ids(self):
        ear_ids = self.datareader.query.specification_based_ids(self._specification, exclude_subjects=self._exclude_ids)
        subject_ids, _ = zip(*ear_ids)
        return tuple(sorted(set([str_id for str_id in subject_ids if isinstance(str_id, str)])) +
                     sorted(set([int_id for int_id in subject_ids if isinstance(int_id, int)])))


    @property
    def preprocessing_chain(self):
        return self._specification['hrir']['preprocess']


    @property
    def transform_chain(self):
        return self._specification['hrir']['transform']


    @property
    def full_chain(self):
        return self._specification['hrir']['preprocess'] + self._specification['hrir']['transform']


    def verify_chain(self):
        chain_length = len(self.full_chain)
        plane_positions = [i for i, t in enumerate(self.full_chain) if isinstance(t, PlaneTransform)]
        grid_positions = [i for i, t in enumerate(self.full_chain) if isinstance(t, GriddedPositionsTransform)]
        domain_positions = [i for i, t in enumerate(self.full_chain) if isinstance(t, DomainTransform) and t.domain != 'time']
        freefield_positions = [i for i, t in enumerate(self.full_chain) if isinstance(t, CompensateFreefieldTransform)]
        # Only one (non-time) domain transformation can take place
        if len(domain_positions) > 1:
            raise RuntimeError('At most one domain transformation is allowed per processing chain.')
        # Only one freefield compensation can take place
        if len(freefield_positions) > 1:
            raise RuntimeError('At most one freefield compensation is allowed per processing chain.')
        # Plane selection or flattening needs to be done last
        if len(plane_positions) != 0 and plane_positions[0] < chain_length - 1:
            raise RuntimeError('A PlaneTransform can only be the last transform in a processing chain.')
        if len(grid_positions) != 0 and grid_positions[0] < chain_length - 1:
            raise RuntimeError('A GriddedPositionsTransform can only be the last transform in a processing chain.')
        # Freefield compensation can only be done in time domain
        if len(freefield_positions) != 0 and len(domain_positions) != 0 and freefield_positions[0] > domain_positions[-1]:
            raise RuntimeError('Freefield compensation needs to be done in the time domain, so move this transform before the domain transform.')


    def insert_transform(self, spec_type: type[Spec], transform_callable: Callable, index: int):
        if spec_type.name not in self._specification.keys():
            raise ValueError(f'No {spec_type.name.title()}Spec in this Dataset.')
        self._specification[spec_type.name]['transform'].insert(index, transform_callable)
        self.verify_chain()


    def prepend_transform(self, spec_type: type[Spec], transform_callable: Callable):
        self.insert_transform(spec_type, transform_callable, 0)


    def append_transform(self, spec_type: type[Spec], transform_callable: Callable):
        self.insert_transform(spec_type, transform_callable, len(self._specification[spec_type.name]['transform']))


    def delete_transform(self, spec_type: type[Spec], index: int):
        self._specification[spec_type.name]['transform'].pop(index)


    @property
    def hrir_length(self):
        length = self._hrir_length
        for t in self.full_chain:
            if isinstance(t, ResampleTransform):
                length = round(length * t.target_rate / t.original_rate)
            elif isinstance(t, TruncateTransform) and t._truncated_length is not None:
                length = t._truncated_length
            elif isinstance(t, PadTransform):
                length += sum(t._pad_width)
            elif isinstance(t, CompensateFreefieldTransform):
                if t.truncated_samples is not None:
                    length = t.truncated_samples
                else:
                    length += t.freefield_length - 1
                length += t.append_samples
        return length


    @property
    def hrtf_frequencies(self):
        if self._specification['hrir'].get('domain', 'time') == 'time':
            return self._hrtf_frequencies
        region_selector = [t for t in self.full_chain if isinstance(t, SelectValueRangeTransform)]
        if not region_selector:
            return self._hrtf_frequencies
        return self._hrtf_frequencies[region_selector[0]._selection]


    def positions(self, coordinate_system=None):
        if not self.grid_mask.size:
            return np.array([])
        req_coordinates = self._coordinate_transform(coordinate_system, *np.meshgrid(self.fundamental_angles, self.orthogonal_angles, self.radii, indexing='ij'))
        coordinates_grid = np.ma.MaskedArray(
            np.stack(req_coordinates, axis=-1),
            np.broadcast_to(~self.grid_mask[..., np.newaxis], (*self.grid_mask.shape, 3)),
        )
        if self.full_chain:
            if isinstance(self.full_chain[-1], GriddedPositionsTransform):
                return coordinates_grid
            elif isinstance(self.full_chain[-1], PlaneTransform):
                return self.full_chain[-1](coordinates_grid)
        return coordinates_grid.compressed().reshape(-1, 3)


    @property
    def plane_angle_name(self):
        try:
            return self._plane_transform.plane_angle_name
        except AttributeError:
            return


    @property
    def plane_angles(self):
        try:
            return self._plane_transform.plane_angles
        except AttributeError:
            return


    @property
    def positive_angles(self):
        try:
            return self._plane_transform.positive_angles
        except AttributeError:
            return


    @positive_angles.setter
    def positive_angles(self, value):
        try:
            self._plane_transform.positive_angles = value
        except AttributeError:
            return


    @property
    def min_angle(self):
        try:
            return self._plane_transform.min_angle
        except AttributeError:
            return


    @property
    def max_angle(self):
        try:
            return self._plane_transform.max_angle
        except AttributeError:
            return


    def plot_plane(self, idx, ax=None, vmin=None, vmax=None, title=None, lineplot=False, cmap='viridis', continuous=False, colorbar=True, log_freq=False, plane=None, offset=None, positive_angles=None, distance: None | str | Real = None):
        if not isinstance(distance, Real) and distance not in ('nearest', 'farthest', None):
            raise ValueError(f'Invalid distance "{distance}". Only "nearest", "farthest" or a value in meter is allowed.')
        if self._specification.get('hrir') is None:
            raise ValueError('No HRIR data in this Dataset.')
        loaded_plane = self._specification['hrir'].get('plane')
        if loaded_plane is not None:
            # Planar dataset
            if plane is not None and plane != loaded_plane:
                raise ValueError(f'This Dataset only contains a {loaded_plane} plane, so requesting a {plane} is invalid.')
            loaded_offset = self._specification['hrir']['plane_offset']
            if offset is not None and offset != loaded_offset :
                raise ValueError(f'This Dataset only contains a {loaded_plane} plane with offset {loaded_offset}, so requesting an offset of {offset} is invalid.')
            if distance is not None:
                req_distance = self._specification['hrir'].get('distance')
                if isinstance(distance, str) and distance != req_distance:
                    if isinstance(req_distance, str):
                        raise ValueError(f'This Dataset only contains positions at the {req_distance} distance, so requesting the {distance} distance is invalid.')
                    else:
                        raise ValueError(f'This Dataset only contains positions at {req_distance}m distance, so requesting the {distance} distance is invalid.')
                loaded_distance = self.radii[0]
                if isinstance(distance, Real) and quantise(distance, self.datareader._distance_quantisation) != loaded_distance:
                    raise ValueError(f'This Dataset only contains positions at {loaded_distance}m distance, so requesting a distance of {distance}m is invalid.')
            plane = self._specification['hrir']['plane']
            offset = self._specification['hrir']['plane_offset']
        else:
            # Non-planar dataset
            if plane is None:
                raise ValueError('A plane to plot needs to be specified for non-planar Datasets.')
            if offset is None:
                offset = 0
            if positive_angles is None:
                positive_angles = False
            grid_transform = GriddedPositionsTransform(self.grid_mask)
            req_fundamental_angles, req_orthogonal_angles = self.datareader.PlaneTransform.convert_plane_angles(plane, None, offset)

            if isinstance(distance, Real):
                req_distance = quantise(distance, self.datareader._distance_quantisation)
                distance_mask = np.isin(self.radii, req_distance)
            else:
                distance_mask = np.zeros_like(self.radii, dtype=bool)
                if distance == 'nearest':
                    distance_mask[0] = True
                else:
                    distance_mask[-1] = True

            fundamental_mask = (np.ones_like(self.fundamental_angles, dtype=bool) if req_fundamental_angles is None
                                else np.isin(self.fundamental_angles, req_fundamental_angles))
            orthogonal_mask = (np.ones_like(self.orthogonal_angles, dtype=bool) if req_orthogonal_angles is None
                               else np.isin(self.orthogonal_angles, req_orthogonal_angles))
            subset_mask = self.grid_mask & fundamental_mask[:, None, None] & orthogonal_mask[None, :, None] & distance_mask[None, None, :]
            if not subset_mask.any():
                raise ValueError(f'There are no measurements at {distance}{'m' if isinstance(distance, Real) else ''} distance in the {plane} plane.')
            plane_fundamental_mask = subset_mask.any(axis=(1, 2))
            plane_orthogonal_mask = subset_mask.any(axis=(0, 2))
            plane_distance_mask = subset_mask.any(axis=(0, 1))
            plane_grid_mask = self.grid_mask[np.ix_(plane_fundamental_mask, plane_orthogonal_mask, plane_distance_mask)]

            plane_fundamental_angles = self.fundamental_angles[plane_fundamental_mask]
            plane_orthogonal_angles = self.orthogonal_angles[plane_orthogonal_mask]
            plane_radius = self.radii[plane_distance_mask].item()

        hrir_role = 'features' if 'hrir' in self._features_keys else 'target' if 'hrir' in self._target_keys else 'group'
        if vmin is None or vmax is None:
            all_subjects = self[:][hrir_role]
            if loaded_plane is None:
                all_subjects = grid_transform(all_subjects)[:, subset_mask]
            if vmin is None:
                vmin = all_subjects.min()
            if vmax is None:
                vmax = all_subjects.max()
        data = self[idx][hrir_role]
        if loaded_plane is None:
            gridded_data = grid_transform(data)
            plane_gridded_shape = (*plane_grid_mask.shape, data.shape[-1])
            plane_gridded_data = np.ma.masked_all(plane_gridded_shape, dtype=gridded_data.dtype)
            plane_gridded_data[plane_grid_mask] = gridded_data[subset_mask]
            plane_transform_subset = self.datareader.PlaneTransform(
                plane, offset, positive_angles,
                plane_fundamental_angles, plane_orthogonal_angles, plane_grid_mask
            )
            data = plane_transform_subset(plane_gridded_data)
            plane_angles = plane_transform_subset.plane_angles
            plane_angle_name = plane_transform_subset.plane_angle_name
        else:
            plane_angles = self.plane_angles
            plane_angle_name = self.plane_angle_name

        if self._specification['hrir']['domain'] == 'time':
            if lineplot:
                ax = plot_hrir_lines(data, plane_angles, plane_angle_name, self.hrir_samplerate, ax=ax, vmin=vmin, vmax=vmax)
            else:
                ax = plot_hrir_plane(data, plane_angles, plane_angle_name, self.hrir_samplerate, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, continuous=continuous, colorbar=colorbar)
        else:
            if lineplot:
                ax = plot_hrtf_lines(data, plane_angles, plane_angle_name, self.hrtf_frequencies, log_freq=log_freq, ax=ax, vmin=vmin, vmax=vmax)
            else:
                ax = plot_hrtf_plane(data, plane_angles, plane_angle_name, self.hrtf_frequencies, log_freq=log_freq, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, continuous=continuous, colorbar=colorbar)

        if title is None:
            title = "{} Plane{} of Subject {}'s {} Ear{}".format(plane.title(),
                ' With Offset {}°'.format(offset) if offset != 0 or plane in ('vertical', 'interaural') else '',
                self.subject_ids[idx],
                self.sides[idx].replace('-', ' ').title(),
                ' at {}m Distance'.format(plane_radius) if len(self.radii) > 1 else '',
            )
        ax.set_title(title)
        return ax


    def plot_positions(self, ax=None, title=None, limit=None, **kwargs):
        if self._specification.get('hrir') is None:
            raise ValueError('No HRIR data in this Dataset.')
        plane = self._specification['hrir'].get('plane')
        if plane is not None:
            # Planar Dataset
            if plane in ('horizontal', 'interaural', 'frontal'):
                zero_location = 'N'
                direction = 'counterclockwise'
            else: # median or vertical
                zero_location = 'W'
                direction = 'clockwise'
            ax = plot_plane_positions(self.plane_angles, self.min_angle, self.max_angle, True, self.radii, zero_location, direction, ax, limit, **kwargs)
            if title is None:
                plane_offset = self._specification['hrir']['plane_offset']
                title = "Source Positions in {}'s {} Plane{}".format(self.datareader.query.collection_id().title(), plane.title(),
                    ' With Offset {}°'.format(plane_offset) if plane_offset != 0 or plane in ('vertical', 'interaural') else ''
                )
        else:
            # Non-planar Dataset
            ax = plot_3d_positions(self.positions(coordinate_system='cartesian'), ax=ax, ax_limit=limit, **kwargs)
            if title is None:
                title = f'Source Positions in {self.datareader.query.collection_id().title()}'
        ax.set_title(title)
        return ax


def split_by_angles(dataset: Dataset):
    angle_datasets = []
    for row_idx, fundamental_angle in enumerate(dataset.fundamental_angles):
        for column_idx, orthogonal_angle in enumerate(dataset.orthogonal_angles):
            for radius_idx, radius in enumerate(dataset.radii):
                if dataset.grid_mask[row_idx, column_idx, radius_idx].item():
                    angle_dataset = deepcopy(dataset)
                    angle_dataset.fundamental_angles = np.array([fundamental_angle])
                    angle_dataset.orthogonal_angles = np.array([orthogonal_angle])
                    angle_dataset.radii = np.array([radius])
                    angle_dataset.grid_mask = np.array([True])
                    try:
                        # Trigger recalculation of plane angles in planar datasets
                        angle_dataset.positive_angles = dataset.positive_angles
                    except AttributeError:
                        pass
                    angle_dataset._data['hrir'] = angle_dataset._data['hrir'][:, row_idx:row_idx+1, column_idx:column_idx+1, radius_idx:radius_idx+1]
                    angle_datasets.append(angle_dataset)
    return angle_datasets


def _get_value_from_hrir_spec(key, *specs):
    hrir_dict = {k: v for spec in sanitise_multiple_specs(specs) if isinstance(spec, HrirSpec) for k, v in spec.spec.items()}
    return hrir_dict.get(key)


class Cipic(Dataset):
    """CIPIC HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        features_spec: Union[Spec, Iterable[Spec]],
        target_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        group_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        if _get_value_from_hrir_spec('variant', features_spec, target_spec, group_spec) is not None:
            raise ValueError('Only the default hrir_variant (`None`) is available.')
        datareader = CipicDataReader(
            anthropometry_matfile_path=Path(root)/'anthropometry/anthro.mat',
            sofa_directory_path=Path(root)/'sofa',
            image_directory_path=Path(root)/'ear_photos',
            download=download,
            verify=verify,
        )
        super().__init__(datareader, features_spec, target_spec, group_spec, subject_ids, subject_requirements, exclude_ids, dtype)


class Ari(Dataset):
    """ARI HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        features_spec: Union[Spec, Iterable[Spec]],
        target_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        group_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        if _get_value_from_hrir_spec('variant', features_spec, target_spec, group_spec) is not None:
            raise ValueError('Only the default hrir_variant (`None`) is available.')
        datareader = AriDataReader(
            anthropometry_matfile_path=Path(root)/'anthro.mat',
            sofa_directory_path=Path(root)/'sofa',
            download=download,
            verify=verify,
        )
        super().__init__(datareader, features_spec, target_spec, group_spec, subject_ids, subject_requirements, exclude_ids, dtype)


class Listen(Dataset):
    """Listen HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        features_spec: Union[Spec, Iterable[Spec]],
        target_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        group_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        datareader = ListenDataReader(
            sofa_directory_path=Path(root)/'sofa',
            anthropometry_directory_path=Path(root)/'anthropometry',
            hrir_variant=_get_value_from_hrir_spec('variant', features_spec, target_spec, group_spec),
            download=download,
            verify=verify,
        )
        super().__init__(datareader, features_spec, target_spec, group_spec, subject_ids, subject_requirements, exclude_ids, dtype)


class CrossMod(Dataset):
    """CrossMod HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        features_spec: Union[Spec, Iterable[Spec]],
        target_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        group_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        datareader = CrossModDataReader(
            sofa_directory_path=Path(root)/'sofa',
            hrir_variant=_get_value_from_hrir_spec('variant', features_spec, target_spec, group_spec),
            download=download,
            verify=verify,
        )
        super().__init__(datareader, features_spec, target_spec, group_spec, subject_ids, subject_requirements, exclude_ids, dtype)


class BiLi(Dataset):
    """BiLi HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        features_spec: Union[Spec, Iterable[Spec]],
        target_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        group_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        datareader = BiLiDataReader(
            sofa_directory_path=Path(root)/'sofa',
            hrir_variant=_get_value_from_hrir_spec('variant', features_spec, target_spec, group_spec),
            hrir_samplerate=_get_value_from_hrir_spec('samplerate', features_spec, target_spec, group_spec),
            download=download,
            verify=verify,
        )
        super().__init__(datareader, features_spec, target_spec, group_spec, subject_ids, subject_requirements, exclude_ids, dtype)


class Ita(Dataset):
    """ITA HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        features_spec: Union[Spec, Iterable[Spec]],
        target_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        group_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        if _get_value_from_hrir_spec('variant', features_spec, target_spec, group_spec) is not None:
            raise ValueError('Only the default hrir_variant (`None`) is available.')
        datareader = ItaDataReader(
            sofa_directory_path=Path(root)/'sofa',
            anthropometry_csvfile_path=Path(root)/'Dimensions.xlsx',
            download=download,
            verify=verify,
        )
        super().__init__(datareader, features_spec, target_spec, group_spec, subject_ids, subject_requirements, exclude_ids, dtype)


class Hutubs(Dataset):
    """HUTUBS HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        features_spec: Union[Spec, Iterable[Spec]],
        target_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        group_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        if _get_value_from_hrir_spec('variant', features_spec, target_spec, group_spec) is not None:
            raise ValueError('Only the default hrir_variant (`None`) is available.')
        datareader = HutubsDataReader(
            sofa_directory_path=Path(root)/'sofa',
            anthropometry_csvfile_path=Path(root)/'AntrhopometricMeasures.csv',
            hrir_method=_get_value_from_hrir_spec('method', features_spec, target_spec, group_spec),
            download=download,
            verify=verify,
            )
        super().__init__(datareader, features_spec, target_spec, group_spec, subject_ids, subject_requirements, exclude_ids, dtype)


class Riec(Dataset):
    """RIEC HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        features_spec: Union[Spec, Iterable[Spec]],
        target_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        group_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        if _get_value_from_hrir_spec('variant', features_spec, target_spec, group_spec) is not None:
            raise ValueError('Only the default hrir_variant (`None`) is available.')
        datareader = RiecDataReader(
            sofa_directory_path=Path(root)/'sofa',
            download=download,
            verify=verify,
        )
        super().__init__(datareader, features_spec, target_spec, group_spec, subject_ids, subject_requirements, exclude_ids, dtype)


class Chedar(Dataset):
    """CHEDAR HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        features_spec: Union[Spec, Iterable[Spec]],
        target_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        group_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        if _get_value_from_hrir_spec('variant', features_spec, target_spec, group_spec) is not None:
            raise ValueError('Only the default hrir_variant (`None`) is available.')
        datareader = ChedarDataReader(
            sofa_directory_path=Path(root)/'sofa',
            anthropometry_matfile_path=Path(root)/'measurements.mat',
            distance=_get_value_from_hrir_spec('distance', features_spec, target_spec, group_spec),
            download=download,
            verify=verify,
        )
        super().__init__(datareader, features_spec, target_spec, group_spec, subject_ids, subject_requirements, exclude_ids, dtype)


class Widespread(Dataset):
    """Widespread HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        features_spec: Union[Spec, Iterable[Spec]],
        target_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        group_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        grid: str = 'UV',
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        if _get_value_from_hrir_spec('variant', features_spec, target_spec, group_spec) is not None:
            raise ValueError('Only the default hrir_variant (`None`) is available.')
        datareader = WidespreadDataReader(
            sofa_directory_path=Path(root)/'sofa',
            distance=_get_value_from_hrir_spec('distance', features_spec, target_spec, group_spec),
            grid=grid,
            download=download,
            verify=verify,
        )
        super().__init__(datareader, features_spec, target_spec, group_spec, subject_ids, subject_requirements, exclude_ids, dtype)


class Sadie2(Dataset):
    """SADIE II HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        features_spec: Union[Spec, Iterable[Spec]],
        target_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        group_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        if _get_value_from_hrir_spec('variant', features_spec, target_spec, group_spec) is not None:
            raise ValueError('Only the default hrir_variant (`None`) is available.')
        datareader = Sadie2DataReader(
            sofa_directory_path=Path(root)/'Database-Master_V1-4',
            image_directory_path=Path(root)/'Database-Master_V1-4',
            hrir_samplerate=_get_value_from_hrir_spec('samplerate', features_spec, target_spec, group_spec),
            download=download,
            verify=verify,
        )
        super().__init__(datareader, features_spec, target_spec, group_spec, subject_ids, subject_requirements, exclude_ids, dtype)


class Princeton3D3A(Dataset):
    """3D3A HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        features_spec: Union[Spec, Iterable[Spec]],
        target_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        group_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        datareader = Princeton3D3ADataReader(
            sofa_directory_path=Path(root)/'HRTFs',
            anthropometry_directory_path=Path(root)/'Anthropometric-Data',
            hrir_method=_get_value_from_hrir_spec('method', features_spec, target_spec, group_spec),
            hrir_variant=_get_value_from_hrir_spec('variant', features_spec, target_spec, group_spec),
            download=download,
            verify=verify,
        )
        super().__init__(datareader, features_spec, target_spec, group_spec, subject_ids, subject_requirements, exclude_ids, dtype)


class Scut(Dataset):
    """SCUT HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        features_spec: Union[Spec, Iterable[Spec]],
        target_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        group_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        if _get_value_from_hrir_spec('variant', features_spec, target_spec, group_spec) is not None:
            raise ValueError('Only the default hrir_variant (`None`) is available.')
        datareader = ScutDataReader(
            sofa_directory_path=Path(root)/'sofa',
            anthropometry_csvfile_path=Path(root)/'AnthropometricParameters.csv',
            download=download,
            verify=verify,
        )
        super().__init__(datareader, features_spec, target_spec, group_spec, subject_ids, subject_requirements, exclude_ids, dtype)


class Sonicom(Dataset):
    """SONICOM HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        features_spec: Union[Spec, Iterable[Spec]],
        target_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        group_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        datareader = SonicomDataReader(
            sofa_directory_path=Path(root),
            hrir_variant=_get_value_from_hrir_spec('variant', features_spec, target_spec, group_spec),
            hrir_samplerate=_get_value_from_hrir_spec('samplerate', features_spec, target_spec, group_spec),
            download=download,
            verify=verify,
        )
        super().__init__(datareader, features_spec, target_spec, group_spec, subject_ids, subject_requirements, exclude_ids, dtype)


class MitKemar(Dataset):
    """MIT KEMAR HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        features_spec: Union[Spec, Iterable[Spec]],
        target_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        group_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        if _get_value_from_hrir_spec('variant', features_spec, target_spec, group_spec) is not None:
            raise ValueError('Only the default hrir_variant (`None`) is available.')
        datareader = MitKemarDataReader(
            sofa_directory_path=Path(root)/'sofa',
            download=download,
            verify=verify,
        )
        super().__init__(datareader, features_spec, target_spec, group_spec, subject_ids, subject_requirements, exclude_ids, dtype)


class CustomSphericalDataset(Dataset):
    """Custom HRTF Dataset
    """
    def __init__(
        self,
        collection_id: str,
        file_paths: Iterable[Union[str, Path]],
        features_spec: Union[Spec, Iterable[Spec]],
        target_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        group_spec: Optional[Union[Spec, Iterable[Spec]]] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        if _get_value_from_hrir_spec('variant', features_spec, target_spec, group_spec) is not None:
            raise ValueError('Only the default hrir_variant (`None`) is available.')
        datareader = CustomSphericalDataReader(collection_id, file_paths)
        super().__init__(datareader, features_spec, target_spec, group_spec, subject_ids, subject_requirements, exclude_ids, dtype)
