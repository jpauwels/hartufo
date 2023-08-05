from .datareader import DataReader, CipicDataReader, AriDataReader, ListenDataReader, BiLiDataReader, ItaDataReader, HutubsDataReader, RiecDataReader, ChedarDataReader, WidespreadDataReader, Sadie2DataReader, Princeton3D3ADataReader, ScutDataReader, SonicomDataReader
from .transforms.hrirs import BatchTransform, ScaleTransform, MinPhaseTransform, ResampleTransform, TruncateTransform, DomainTransform
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from numbers import Integral
from typing import Dict, Iterable, Optional
import numpy as np
import numpy.typing as npt
from scipy.fft import rfftfreq


class Dataset:

    def __init__(
        self,
        datareader: DataReader,
        features_spec: Dict,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: npt.DTypeLike = np.float32,
        dict_format = True,
    ) -> None:
        super().__init__()
        self._query = datareader.query
        self.fundamental_angle_name = datareader.fundamental_angle_name
        self.orthogonal_angle_name = datareader.orthogonal_angle_name
        # Allow specifying ids that are excluded by default without explicitly overriding `exclude_ids``
        if subject_ids is not None and not isinstance(subject_ids, str) and exclude_ids is None:
            exclude_ids = ()
        self._exclude_ids = exclude_ids
        self._dict_format = dict_format
        self.dtype = dtype

        if target_spec is None:
            target_spec = {}
        if group_spec is None:
            group_spec = {}

        self._specification = {**features_spec, **target_spec, **group_spec}
        if not self._specification:
            raise ValueError('At least one specification should not be empty')
        self._features_keys = tuple(features_spec.keys())
        self._target_keys = tuple(target_spec.keys())
        self._group_keys = tuple(group_spec.keys())

        if subject_requirements is not None:
            self._specification = {**self._specification, **subject_requirements}
        ear_ids = self._query.specification_based_ids(self._specification, include_subjects=subject_ids, exclude_subjects=exclude_ids)

        if len(ear_ids) == 0:
            if len(self._query.specification_based_ids(self._specification)) == 0:
                raise ValueError('Empty dataset. Check if its configuration and paths are correct.')
            if subject_ids:
                raise ValueError('None of the explicitly requested subject IDs are available.')
            self.subject_ids = tuple()
            self.hrir_samplerate = None
            self.hrir_length = None
            self.hrtf_frequencies = None
            self.fundamental_angles = np.array([])
            self.orthogonal_angles = np.array([])
            self.radii = np.array([])
            self._selection_mask = None
            self._data = {}
            return

        self.subject_ids, self.sides = zip(*ear_ids)

        for k in self._specification.keys():
            transform_callable = self._specification[k].get('transform', [])
            if callable(transform_callable):
                transform_callable = [transform_callable]
            self._specification[k]['transform'] = transform_callable
            preprocess_callable = self._specification[k].get('preprocess', [])
            if callable(preprocess_callable):
                preprocess_callable = [preprocess_callable]
            self._specification[k]['preprocess'] = preprocess_callable
                
        if 'hrirs' in self._specification.keys():
            # Verify angle symmetry before reading complete dataset
            self.fundamental_angles, self.orthogonal_angles, self.radii, self._selection_mask, *_ = datareader._map_sofa_position_order_to_matrix(
                self.subject_ids[0],
                self._specification['hrirs'].get('fundamental_angles'),
                self._specification['hrirs'].get('orthogonal_angles'),
                self._specification['hrirs'].get('distance'),
            )
            if self._specification['hrirs'].get('side', '').startswith('both-'):
                datareader._verify_angle_symmetry(self.fundamental_angles, self.orthogonal_angles)
            # Construct HRIR processing pipeline
            hrir_pipeline = []
            recorded_hrir_length = datareader.hrir_length(self.subject_ids[0])
            recorded_samplerate = datareader.hrir_samplerate(self.subject_ids[0])
            if self._specification['hrirs'].get('additive_scale_factor') is not None or self._specification['hrirs'].get('multiplicative_scale_factor'):
                hrir_pipeline.append(ScaleTransform(self._specification['hrirs'].get('additive_scale_factor'), self._specification['hrirs'].get('multiplicative_scale_factor')))
            if self._specification['hrirs'].get('min_phase', False):
                hrir_pipeline.append(MinPhaseTransform(recorded_hrir_length))
            if self._specification['hrirs'].get('samplerate') is None:
                self.hrir_samplerate = recorded_samplerate
            else:
                self.hrir_samplerate = self._specification['hrirs']['samplerate']
                hrir_pipeline.append(ResampleTransform(self.hrir_samplerate / recorded_samplerate))
            if self._specification['hrirs'].get('length') is None:
                self.hrir_length = recorded_hrir_length
            else:
                self.hrir_length = self._specification['hrirs']['length']
                hrir_pipeline.append(TruncateTransform(self.hrir_length))
            if self._specification['hrirs'].get('domain') is not None or self.dtype is not None:
                hrir_pipeline.append(DomainTransform(self._specification['hrirs'].get('domain', 'time'), self.dtype))
            self._specification['hrirs']['preprocess'] = hrir_pipeline + self._specification['hrirs']['preprocess']
            self.hrtf_frequencies = rfftfreq(self.hrir_length, 1/self.hrir_samplerate)

        self._data = defaultdict(list)
        for subject, side in ear_ids:
            if 'image' in self._specification.keys():
                self._data['image'].append(datareader.image(subject, side=side, rear=self._specification['image'].get('rear', False)))
            if 'anthropometry' in self._specification.keys():
                self._data['anthropometry'].append(datareader.anthropometric_data(subject, side=side, select=self._specification['anthropometry'].get('select')).astype(self.dtype))
            if 'hrirs' in self._specification.keys():
                self._data['hrirs'].append(datareader.hrir(subject, side, self._specification['hrirs'].get('fundamental_angles'), self._specification['hrirs'].get('orthogonal_angles'), self._specification['hrirs'].get('distance')))
            if 'subject' in self._specification.keys():
                self._data['subject'].append(subject)
            if 'side' in self._specification.keys():
                self._data['side'].append(side)
            if 'collection' in self._specification.keys():
                self._data['collection'].append(datareader.query.collection_id)

        for k in self._specification.keys():
            if k in ('hrirs', 'anthropometry'):
                try:
                    self._data[k] = np.ma.stack(self._data[k])
                except ValueError:
                    raise ValueError(f'Not all data points have the same {k} shape') from None
            for preprocess in self._specification[k]['preprocess']:
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
        ear_ids = self._query.specification_based_ids(self._specification, exclude_subjects=self._exclude_ids)
        subject_ids, _ = zip(*ear_ids)
        return tuple(np.unique(subject_ids))


    def insert_transform(self, data_type, transform_callable, index):
        if data_type not in self._specification.keys():
            raise ValueError(f'This Dataset does not contain any {data_type}.')
        self._specification[data_type]['transform'].insert(index, transform_callable)


    def prepend_transform(self, data_type, transform_callable):
        self.insert_transform(data_type, transform_callable, 0)


    def append_transform(self, data_type, transform_callable):
        self.insert_transform(data_type, transform_callable, len(self._specification[data_type]['transform']))


def split_by_angles(dataset: Dataset):
    angle_datasets = []
    for row_idx, fundamental_angle in enumerate(dataset.fundamental_angles):
        for column_idx, orthogonal_angle in enumerate(dataset.orthogonal_angles):
            for radius_idx, radius in enumerate(dataset.radii):
                if not dataset._selection_mask[row_idx, column_idx, radius_idx].item():
                    angle_dataset = deepcopy(dataset)
                    angle_dataset.fundamental_angles = np.array([fundamental_angle])
                    angle_dataset.orthogonal_angles = np.array([orthogonal_angle])
                    angle_dataset.radii = np.array([radius])
                    angle_dataset._selection_mask = np.array([False])
                    try:
                        # Trigger recalculation of plane angles in planar datasets
                        angle_dataset.positive_angles = dataset.positive_angles
                    except AttributeError:
                        pass
                    angle_dataset._data['hrirs'] = angle_dataset._data['hrirs'][:, row_idx:row_idx+1, column_idx:column_idx+1, radius_idx:radius_idx+1]
                    angle_datasets.append(angle_dataset)
    return angle_datasets


def _get_hrir_samplerate_from_spec(features_spec, target_spec, group_spec):
    spec_list = [spec for spec in (features_spec, target_spec, group_spec) if spec is not None]
    hrir_spec = {k: v for d in spec_list for k, v in d.items()}.get('hrirs', {})
    return hrir_spec.get('samplerate')


def _get_hrir_distance_from_spec(features_spec, target_spec, group_spec):
    spec_list = [spec for spec in (features_spec, target_spec, group_spec) if spec is not None]
    hrir_spec = {k: v for d in spec_list for k, v in d.items()}.get('hrirs', {})
    return hrir_spec.get('distance')


class Cipic(Dataset):
    """CIPIC HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        features_spec: Dict,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
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
        features_spec: Dict,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
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
        features_spec: Dict,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        hrtf_type: str = 'compensated',
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        datareader = ListenDataReader(
            sofa_directory_path=Path(root)/'sofa',
            anthropometry_directory_path=Path(root)/'anthropometry',
            hrtf_type=hrtf_type,
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
        features_spec: Dict,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        hrtf_type: str = 'compensated',
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        datareader = BiLiDataReader(
            sofa_directory_path=Path(root)/'sofa',
            hrtf_type=hrtf_type,
            hrir_samplerate=_get_hrir_samplerate_from_spec(features_spec, target_spec, group_spec),
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
        features_spec: Dict,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
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
        features_spec: Dict,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        measured_hrtf: bool = True,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        datareader = HutubsDataReader(
            sofa_directory_path=Path(root)/'sofa',
            anthropometry_csvfile_path=Path(root)/'AntrhopometricMeasures.csv',
            measured_hrtf=measured_hrtf,
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
        features_spec: Dict,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
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
        features_spec: Dict,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        datareader = ChedarDataReader(
            sofa_directory_path=Path(root)/'sofa',
            anthropometry_matfile_path=Path(root)/'measurements.mat',
            distance=_get_hrir_distance_from_spec(features_spec, target_spec, group_spec),
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
        features_spec: Dict,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        grid: str = 'UV',
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        datareader = WidespreadDataReader(
            sofa_directory_path=Path(root)/'sofa',
            distance=_get_hrir_distance_from_spec(features_spec, target_spec, group_spec),
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
        features_spec: Dict,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        datareader = Sadie2DataReader(
            sofa_directory_path=Path(root)/'Database-Master_V1-4',
            image_directory_path=Path(root)/'Database-Master_V1-4',
            hrir_samplerate=_get_hrir_samplerate_from_spec(features_spec, target_spec, group_spec),
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
        features_spec: Dict,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        hrtf_method: str = 'measured',
        hrtf_type: str = 'compensated',
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        datareader = Princeton3D3ADataReader(
            sofa_directory_path=Path(root)/'HRTFs',
            anthropometry_directory_path=Path(root)/'Anthropometric-Data',
            hrtf_method=hrtf_method,
            hrtf_type=hrtf_type,
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
        features_spec: Dict,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
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
        features_spec: Dict,
        target_spec: Optional[Dict] = None,
        group_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[int]] = None,
        hrtf_type: str = 'compensated',
        dtype: type = np.float32,
        download: bool = False,
        verify: bool = False,
    ) -> None:
        datareader = SonicomDataReader(
            sofa_directory_path=Path(root),
            hrtf_type=hrtf_type,
            hrir_samplerate=_get_hrir_samplerate_from_spec(features_spec, target_spec, group_spec),
            download=download,
            verify=verify,
        )
        super().__init__(datareader, features_spec, target_spec, group_spec, subject_ids, subject_requirements, exclude_ids, dtype)
