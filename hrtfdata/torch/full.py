import warnings
from pathlib import Path
from typing import Any, Callable, List, Iterable, Optional, TypeVar, Dict, IO, Tuple, Iterator
import numpy as np
from PIL.Image import Image, LANCZOS
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import ToTensor
# from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from ..datapoint import DataPoint, AriDataPoint, ListenDataPoint, BiLiDataPoint, ItaDataPoint, HutubsDataPoint, RiecDataPoint, ChedarDataPoint, WidespreadDataPoint, Sadie2DataPoint, ThreeDThreeADataPoint



class HRTFDataset(TorchDataset):

    def __init__(
        self,
        datapoint: DataPoint,
        feature_spec: Optional[Dict] = None,
        label_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        image_transform: Optional[Callable] = ToTensor(),
        measurement_transform: Optional[Callable] = None,
        hrir_transform: Optional[Callable] = None,
        # download: bool = True,
    ) -> None:
        super().__init__()#root, transform=transform, target_transform=target_transform) # torchvision dataset
        self._image_transform = image_transform
        self._measurement_transform = measurement_transform
        self._hrir_transform = hrir_transform
        self._query = datapoint.query

        # if download:
        #     self.download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        if feature_spec is None:
            feature_spec = {'images': {'side': 'left', 'rear': False}, 'measurements': {'side': 'left', 'select': 'head-torso', 'partial': False}}
        if label_spec is None:
            label_spec = {'subject': {}}
        

        self._specification = {**feature_spec, **label_spec}
        if subject_requirements is not None:
            self._specification = {**self._specification, **subject_requirements}
        ear_ids = self._query.specification_based_ids(self._specification, include_subjects=subject_ids)

        if len(ear_ids) == 0:
            self.subject_ids = []
            self.hrir_samplerate = None
            self.hrtf_frequencies = None
            self._features = []
            self._targets = []
            self._selected_angles = {}
            return

        self.subject_ids, _ = zip(*ear_ids)

        if 'hrirs' in feature_spec.keys():
            hrir_spec = feature_spec['hrirs']
        elif 'hrirs' in label_spec.keys():
            hrir_spec = label_spec['hrirs']
        else:
            hrir_spec = {}

        self._selected_angles, row_indices, column_indices = datapoint.hrir_angle_indices(
            self.subject_ids[0], hrir_spec.get('row_angles'), hrir_spec.get('column_angles'))

        self.hrir_samplerate = datapoint.hrir_samplerate(self.subject_ids[0])
        self.hrtf_frequencies = datapoint.hrtf_frequencies(self.subject_ids[0])
        hrir_domain = hrir_spec.get('domain', 'time')

        self._features: Any = []
        self._targets: Any = []
        for subject, side in ear_ids:
            for spec, store in (feature_spec, self._features), (label_spec, self._targets):
                subject_data = {}
                if 'images' in spec.keys():
                    subject_data['images'] = datapoint.pinna_image(subject, side=side, rear=spec['images'].get('rear', False))
                if 'measurements' in spec.keys():
                    subject_data['measurements'] = datapoint.anthropomorphic_data(subject, side=side, select=spec['measurements'].get('select', None))
                if 'hrirs' in spec.keys():
                    subject_data['hrirs'] = datapoint.hrir(subject, side=side, domain=hrir_domain, row_indices=row_indices, column_indices=column_indices)
                if 'subject' in spec.keys():
                    subject_data['subject'] = subject
                if 'side' in spec.keys():
                    subject_data['side'] = side
                if 'dataset' in spec.keys():
                    subject_data['dataset'] = datapoint.dataset_id
                store.append(subject_data)


    def __len__(self):
        return len(self._features)


    def __getitem__(self, idx):
        def get_single_item(features, target, group):
            # unify both to simpify on-demand transforms
            characteristics = {**features, **target}

            if 'images' in characteristics:
                width, height = characteristics['images'].size
                resized_im = characteristics['images'].resize((32, 32), resample=LANCZOS, box=(width//2-128, height//2-128, width//2+128, height//2+128)).convert('L')
                if self._image_transform:
                    resized_im = self._image_transform(resized_im)
                # resized_im = resized_im.transpose((1, 2, 0))  # convert to HWC
                characteristics['images'] = resized_im
            if 'measurements' in characteristics and self._measurement_transform:
                characteristics['measurements'] = self._measurement_transform(characteristics['measurements'])
            if 'hrirs' in characteristics and self._hrir_transform:
                characteristics['hrirs'] = self._hrir_transform(characteristics['hrirs'])

            feature_list = [characteristics[k] for k in features.keys()]
            target_list = [characteristics[k] for k in target.keys()]
            features = np.squeeze(np.concatenate(feature_list))
            if len(target_list) > 1:
                target = np.squeeze(np.concatenate(target_list))
            elif len(target_list) == 1:
                target = target_list[0]
            else:
                target = np.array([])
            return {'features': features, 'target': target, 'group': group}

        if isinstance(idx, int):
            return get_single_item(self._features[idx], self._targets[idx], self.subject_ids[idx])
        else:
            items = []
            for features, target, group in zip(self._features[idx], self._targets[idx], self.subject_ids[idx]):
                items.append(get_single_item(features, target, group))
            try:
                return {k: np.stack([d[k] for d in items]) for k in items[0].keys()}
            except ValueError:
                raise ValueError('Not all data points have the same shape')


    @property
    def target_shape(self):
        return np.hstack(tuple(self._targets[0].values())).shape


    @property
    def available_subject_ids(self):
        ear_ids = self._query.specification_based_ids(self._specification)
        subject_ids, _ = zip(*ear_ids)
        return sorted(set(subject_ids))


class ARI(HRTFDataset):
    """ARI HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        feature_spec: Optional[Dict] = None,
        label_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        image_transform: Optional[Callable] = ToTensor(),
        measurement_transform: Optional[Callable] = None,
        hrir_transform: Optional[Callable] = None,
        # download: bool = True,
    ) -> None:
        datapoint = AriDataPoint(
            anthropomorphy_matfile_path=Path(root)/'anthro.mat',
            sofa_directory_path=Path(root)/'sofa',
        )
        super().__init__(datapoint, feature_spec, label_spec, subject_ids, subject_requirements, image_transform, measurement_transform, hrir_transform)


class Listen(HRTFDataset):
    """Listen HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        feature_spec: Optional[Dict] = None,
        label_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        hrir_transform: Optional[Callable] = None,
        # download: bool = True,
    ) -> None:
        datapoint = ListenDataPoint(sofa_directory_path=Path(root)/'sofa/compensated/44100')
        super().__init__(datapoint, feature_spec, label_spec, subject_ids, subject_requirements, None, None, hrir_transform)


class BiLi(HRTFDataset):
    """BiLi HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        feature_spec: Optional[Dict] = None,
        label_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        hrir_transform: Optional[Callable] = None,
        # download: bool = True,
    ) -> None:
        datapoint = BiLiDataPoint(sofa_directory_path=Path(root)/'sofa/compensated/96000')
        super().__init__(datapoint, feature_spec, label_spec, subject_ids, subject_requirements, None, None, hrir_transform)


class ITA(HRTFDataset):
    """ITA HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        feature_spec: Optional[Dict] = None,
        label_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        hrir_transform: Optional[Callable] = None,
        # download: bool = True,
    ) -> None:
        datapoint = ItaDataPoint(sofa_directory_path=Path(root)/'sofa')
        super().__init__(datapoint, feature_spec, label_spec, subject_ids, subject_requirements, None, None, hrir_transform)


class HUTUBS(HRTFDataset):
    """HUTUBS HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        feature_spec: Optional[Dict] = None,
        label_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        hrir_transform: Optional[Callable] = None,
        # download: bool = True,
    ) -> None:
        datapoint = HutubsDataPoint(sofa_directory_path=Path(root)/'sofa')
        super().__init__(datapoint, feature_spec, label_spec, subject_ids, subject_requirements, None, None, hrir_transform)


class RIEC(HRTFDataset):
    """RIEC HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        feature_spec: Optional[Dict] = None,
        label_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        hrir_transform: Optional[Callable] = None,
        # download: bool = True,
    ) -> None:
        datapoint = RiecDataPoint(sofa_directory_path=Path(root)/'sofa')
        super().__init__(datapoint, feature_spec, label_spec, subject_ids, subject_requirements, None, None, hrir_transform)


class CHEDAR(HRTFDataset):
    """CHEDAR HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        feature_spec: Optional[Dict] = None,
        label_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        hrir_transform: Optional[Callable] = None,
        # download: bool = True,
    ) -> None:
        datapoint = ChedarDataPoint(sofa_directory_path=Path(root)/'sofa')
        super().__init__(datapoint, feature_spec, label_spec, subject_ids, subject_requirements, None, None, hrir_transform)


class Widespread(HRTFDataset):
    """Widespread HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        feature_spec: Optional[Dict] = None,
        label_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        hrir_transform: Optional[Callable] = None,
        # download: bool = True,
    ) -> None:
        datapoint = WidespreadDataPoint(sofa_directory_path=Path(root)/'sofa')
        super().__init__(datapoint, feature_spec, label_spec, subject_ids, subject_requirements, None, None, hrir_transform)


class SADIE2(HRTFDataset):
    """SADIE II HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        feature_spec: Optional[Dict] = None,
        label_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        hrir_transform: Optional[Callable] = None,
        # download: bool = True,
    ) -> None:
        datapoint = Sadie2DataPoint(sofa_directory_path=Path(root)/'sofa')
        super().__init__(datapoint, feature_spec, label_spec, subject_ids, subject_requirements, None, None, hrir_transform)


class ThreeDThreeA(HRTFDataset):
    """3D3A HRTF Dataset
    """
    def __init__(
        self,
        root: str,
        feature_spec: Optional[Dict] = None,
        label_spec: Optional[Dict] = None,
        subject_ids: Optional[Iterable[int]] = None,
        subject_requirements: Optional[Dict] = None,
        hrir_transform: Optional[Callable] = None,
        # download: bool = True,
    ) -> None:
        datapoint = ThreeDThreeADataPoint(sofa_directory_path=Path(root)/'sofa')
        super().__init__(datapoint, feature_spec, label_spec, subject_ids, subject_requirements, None, None, hrir_transform)