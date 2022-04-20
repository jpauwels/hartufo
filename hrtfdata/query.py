from abc import abstractmethod
from pathlib import Path
import warnings
import random
import numpy as np
from scipy import io


class DataQuery:

    _default_hrirs_exclude = ()
    _default_images_exclude = ()
    _default_measurements_exclude = ()
    _default_3dmodels_exclude = ()


    def __init__(self, sofa_directory_path=None, mesh_directory_path=None, image_directory_path=None, anthropomorphy_matfile_path=None):
        self.allowed_keys = ['subject', 'side', 'dataset']
        if sofa_directory_path is not None:
            self.sofa_directory_path = Path(sofa_directory_path)
            self.allowed_keys += ['hrirs']
        if mesh_directory_path is not None:
            self.mesh_directory_path = Path(mesh_directory_path)
            self.allowed_keys += ['3d-models']
        if image_directory_path is not None:
            self.image_directory_path = Path(image_directory_path)
            self.allowed_keys += ['images']
        if anthropomorphy_matfile_path is not None:
            self.anthropomorphy_matfile_path = anthropomorphy_matfile_path
            self.anth = io.loadmat(anthropomorphy_matfile_path, squeeze_me=True)
            self.allowed_keys += ['measurements']


    def _all_hrir_ids(self, side):
        return ()


    def _all_mesh_ids(self, side):
        return ()


    def _all_image_ids(self, side, rear):
        return ()


    def _all_measurement_ids(self, side, select, partial):
        return ()


    def validate_specification(self, spec):
        unknown_spec = sorted(set(spec.keys()).difference(self.allowed_keys))
        if unknown_spec:
            raise ValueError(f'Unknown specifier{"s" if len(unknown_spec) > 1 else ""} "{", ".join(unknown_spec)}"')


    def specification_based_ids(self, specification, include_subjects=None, exclude_subjects=None):
        self.validate_specification(specification)
        all_sides = {}
        for key in ('hrirs', 'images', 'measurements'):
            side = specification.get(key, {}).get('side')
            if side is not None:
                all_sides[key] = side
        if len(set(all_sides.values())) > 1:
            warn_strings = [f'{k} ("{v}")' for k, v in all_sides.items()]
            warnings.warn(f"Different sides requested for {', '.join(warn_strings[:-1])} and {warn_strings[-1]}")
        
        if include_subjects is not None and len(include_subjects) == 0:
            return []

        separate_ids = []
        if 'images' in specification.keys():
            separate_ids.append(set(self.image_ids(**{'exclude': exclude_subjects, **specification['images']})))
        if 'measurements' in specification.keys():
            separate_ids.append(set(self.measurements_ids(**{'exclude': exclude_subjects, **specification['measurements']})))
        if 'hrirs' in specification.keys():
            side = specification['hrirs'].get('side')
            exclude = specification['hrirs'].get('exclude', exclude_subjects)
            separate_ids.append(set(self.hrir_ids(side, exclude)))

        ids = sorted(set.intersection(*separate_ids))
        if include_subjects is None:
            return ids
        if len(ids) > 0 and include_subjects == 'first':
            return [ids[0]]
        if len(ids) > 0 and include_subjects == 'last':
            return [ids[-1]]
        if len(ids) > 0 and include_subjects == 'random':
            return [random.choice(ids)]
        return [(i, s) for i, s in ids if i in include_subjects]


    @staticmethod
    def _id_helper(side, id_fn, exclude, default_exclude):
        if side in ['both', 'both-left', 'both-right', None]:
            left_ids = id_fn('left')
            right_ids = id_fn('right')
            if side is not None:
                both_ids = sorted(set(left_ids).intersection(right_ids))
                if side == 'both-left':
                    sides = ('left', 'flipped-right')
                elif side == 'both-right':
                    sides = ('flipped-left', 'right')
                else:
                    sides = ('left', 'right')
                ids = [(i, s) for i in both_ids for s in sides]
            else:
                ids = sorted([(i, 'left') for i in left_ids] + [(i, 'right') for i in right_ids])
        elif side in ['left', 'right']:
            ids = [(i, side) for i in id_fn(side)]
        else:
            raise ValueError(f'Unknown side "{side}"')
        if exclude is None:
            exclude = default_exclude
        return [(i, s) for i, s in ids if i not in exclude]


    def hrir_ids(self, side=None, exclude=None):
        return self._id_helper(side, self._all_hrir_ids, exclude, self._default_hrirs_exclude)


    def image_ids(self, side=None, rear=False, exclude=None):
        if side is None and rear:
            if exclude is None:
                exclude = self._default_images_exclude
            return [(i, s) for i in self._all_image_ids(None, True) for s in ('left', 'right') if i not in exclude]
        return self._id_helper(side, lambda s: self._all_image_ids(s, rear), exclude, self._default_images_exclude)


    def measurements_ids(self, side=None, select=None, partial=False, exclude=None):
        return self._id_helper(side, lambda s: _all_measurement_ids(s, select, partial), exclude, self._default_measurements_exclude)


class CipicDataQuery(DataQuery):

    def __init__(self, sofa_directory_path=None, image_directory_path=None, anthropomorphy_matfile_path=None):
        super().__init__(sofa_directory_path=sofa_directory_path, image_directory_path=image_directory_path, anthropomorphy_matfile_path=anthropomorphy_matfile_path)


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[1]) for x in self.sofa_directory_path.glob('subject_*.sofa')])
    

    @staticmethod
    def _image_suffix(side=None, rear=False):
        return '{}_{}.jpg'.format('_'+side.split('-')[-1] if side else '', 'rear' if rear else 'side')


class AriDataQuery(DataQuery):

    def __init__(self, sofa_directory_path=None, anthropomorphy_matfile_path=None):
        super().__init__(sofa_directory_path=sofa_directory_path, anthropomorphy_matfile_path=anthropomorphy_matfile_path)
        self._default_hrirs_exclude = (10, 22, 826)


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_nh')[1]) for x in self.sofa_directory_path.glob('hrtf [bc]_nh*.sofa')])


class ListenDataQuery(DataQuery):

    def __init__(self, sofa_directory_path):
        super().__init__(sofa_directory_path)


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[1]) for x in self.sofa_directory_path.glob('IRC_????_C_44100.sofa')])


class BiLiDataQuery(DataQuery):

    def __init__(self, sofa_directory_path):
        super().__init__(sofa_directory_path)


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[1]) for x in self.sofa_directory_path.glob('IRC_????_C_HRIR_96000.sofa')])


class ItaDataQuery(DataQuery):

    def __init__(self, sofa_directory_path):
        super().__init__(sofa_directory_path)
        self._default_hrirs_exclude = (2, 14)


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('MRT')[1]) for x in self.sofa_directory_path.glob('MRT??.sofa')])


class HutubsDataQuery(DataQuery):

    def __init__(self, sofa_directory_path):
        super().__init__(sofa_directory_path)


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[0].split('pp')[1]) for x in self.sofa_directory_path.glob('pp??_HRIRs_measured.sofa')])


class RiecDataQuery(DataQuery):

    def __init__(self, sofa_directory_path):
        super().__init__(sofa_directory_path)


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[3]) for x in self.sofa_directory_path.glob('RIEC_hrir_subject_???.sofa')])


class ChedarDataQuery(DataQuery):

    def __init__(self, sofa_directory_path, radius=1):
        super().__init__(sofa_directory_path)
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



    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[1]) for x in self.sofa_directory_path.glob(f'chedar_????_UV{self.radius}.sofa')])


class WidespreadDataQuery(DataQuery):

    def __init__(self, sofa_directory_path, radius=1):
        super().__init__(sofa_directory_path)
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


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[1]) for x in self.sofa_directory_path.glob(f'UV{self.radius}_?????.sofa')])


class Sadie2DataQuery(DataQuery):

    def __init__(self, sofa_directory_path=None, image_directory_path=None):
        super().__init__(sofa_directory_path=sofa_directory_path, image_directory_path=image_directory_path)
        self._default_hrirs_exclude = (1, 2, 3, 4, 5, 6, 7, 8, 9)
        self._default_images_exclude = (3, 16)


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[0][1:]) for x in self.sofa_directory_path.glob('[DH]*/[DH]*_HRIR_SOFA/[DH]*_96K_24bit_512tap_FIR_SOFA.sofa')])


    def _all_image_ids(self, side, rear):
        if rear:
            raise ValueError('No rear pictures available in the SADIE II dataset')
        side_str = 'L' if side == 'left' else 'R'
        return sorted([int(x.stem.split('_')[0].split()[0][1:]) for x in self.image_directory_path.glob(f'[DH]*/[DH]*_Scans/[DH]*[_ ]({side_str}).png')])


class ThreeDThreeADataQuery(DataQuery):

    def __init__(self, sofa_directory_path):
        super().__init__(sofa_directory_path)


    def _all_hrir_ids(self, exclude=()):
        return sorted([int(x.stem.split('_')[0].lstrip('Subject')) for x in self.sofa_directory_path.glob('Subject*_HRIRs.sofa')])


class SonicomDataQuery(DataQuery):

    def __init__(self, sofa_directory_path, samplerate=48000, hrtf_variant='compensated-noitd'):
        super().__init__(sofa_directory_path)
        if samplerate not in (44100, 48000, 96000):
            raise ValueError(f'Sample rate {samplerate} is unavailable. Choose one of 44100, 48000 or 96000.')
        if hrtf_variant == 'compensated':
            self._hrtf_variant_str = 'FreeFieldComp'
        elif hrtf_variant == 'compensated-minphase':
            self._hrtf_variant_str = 'FreeFieldCompMinPhase'
        elif hrtf_variant == 'compensated-noitd':
            self._hrtf_variant_str = 'FreeFieldComp_NoITD'
        elif hrtf_variant == 'compensated-minphase-noitd':
            self._hrtf_variant_str = 'FreeFieldCompMinPhase_NoITD'
        elif hrtf_variant == 'raw':
            self._hrtf_variant_str = 'Raw'
        elif hrtf_variant == 'raw-noitd':
            self._hrtf_variant_str = 'Raw_NoITD'
        elif hrtf_variant == 'windowed':
            self._hrtf_variant_str = 'Windowed'
        elif hrtf_variant == 'windowed-noitd':
            self._hrtf_variant_str = 'Windowed_NoITD'
        else:
            raise ValueError(f'{hrtf_variant}')
        self._samplerate_str = f'{round(samplerate/1000)}kHz'


    def _all_hrir_ids(self, exclude=()):
        return sorted([int(x.stem.split('_')[0].lstrip('P')) for x in self.sofa_directory_path.glob(f'P????/HRTF/{self._samplerate_str}/P????_{self._hrtf_variant_str}_{self._samplerate_str}.sofa')])
