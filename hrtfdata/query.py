from pathlib import Path
import warnings
import random
from numbers import Number
from typing import Optional
import numpy as np
from scipy import io


class DataQuery:

    _default_hrirs_exclude = ()
    _default_images_exclude = ()
    _default_measurements_exclude = ()
    _default_3dmodels_exclude = ()


    def __init__(
        self,
        collection_id: str,
        sofa_directory_path: Optional[str] = None,
        mesh_directory_path: Optional[str] = None,
        image_directory_path: Optional[str] = None,
        anthropomorphy_matfile_path: Optional[str] = None,
        variant_key: str = '',
    ):
        self.collection_id = collection_id
        self.allowed_keys = ['subject', 'side', 'collection']
        if sofa_directory_path is not None:
            self.sofa_directory_path = Path(sofa_directory_path)
            self.allowed_keys += ['hrirs']
        if mesh_directory_path is not None:
            self.mesh_directory_path = Path(mesh_directory_path)
            self.allowed_keys += ['3d-model']
        if image_directory_path is not None:
            self.image_directory_path = Path(image_directory_path)
            self.allowed_keys += ['image']
        if anthropomorphy_matfile_path is not None:
            self.anthropomorphy_matfile_path = anthropomorphy_matfile_path
            self.anth = io.loadmat(anthropomorphy_matfile_path, squeeze_me=True)
            self.allowed_keys += ['anthropometry']
        self._variant_key = variant_key


    def _all_hrir_ids(self, side):
        return ()


    def _all_mesh_ids(self, side):
        return ()


    def _all_image_ids(self, side, rear):
        return ()


    def _all_measurement_ids(self, side, select, partial):
        return ()


    def validate_specification(self, spec):
        def validate_dict(given_dict, allowed_keys, key=''):
            if key:
                try:
                    given_dict = given_dict[key]
                except KeyError:
                    return
            unknown_keys = sorted([x for x in set(given_dict.keys()).difference(allowed_keys) if not isinstance(x, Number)])
            if unknown_keys:
                raise ValueError(f'Unknown specifier{"s" if len(unknown_keys) > 1 else ""} "{", ".join(unknown_keys)}" in {key if key else "specification"}')
        validate_dict(spec, self.allowed_keys)
        validate_dict(spec, ('side', 'domain', 'row_angles', 'column_angles', 'samplerate', 'length', 'min_phase', 'exclude'), 'hrirs')
        validate_dict(spec, (), 'subject')
        validate_dict(spec, (), 'side')
        validate_dict(spec, (), 'collection')
        validate_dict(spec, ('side',), 'images')
        validate_dict(spec, ('side',), 'images')


    def specification_based_ids(self, specification, include_subjects=None, exclude_subjects=None):
        self.validate_specification(specification)
        all_sides = {}
        for key in ('hrirs', 'images', 'anthropometry'):
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
        if 'anthropometry' in specification.keys():
            separate_ids.append(set(self.measurements_ids(**{'exclude': exclude_subjects, **specification['anthropometry']})))
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
                    sides = ('left', 'mirrored-right')
                elif side == 'both-right':
                    sides = ('mirrored-left', 'right')
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
        super().__init__('cipic', sofa_directory_path=sofa_directory_path, image_directory_path=image_directory_path, anthropomorphy_matfile_path=anthropomorphy_matfile_path)


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[1]) for x in self.sofa_directory_path.glob('subject_*.sofa')])
    

    @staticmethod
    def _image_suffix(side=None, rear=False):
        return '{}_{}.jpg'.format('_'+side.split('-')[-1] if side else '', 'rear' if rear else 'side')


class AriDataQuery(DataQuery):

    def __init__(self, sofa_directory_path=None, anthropomorphy_matfile_path=None):
        super().__init__('ari', sofa_directory_path=sofa_directory_path, anthropomorphy_matfile_path=anthropomorphy_matfile_path)
        self._default_hrirs_exclude = (10, 22, 826)


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_nh')[1]) for x in self.sofa_directory_path.glob('hrtf [bc]_nh*.sofa')])


class ListenDataQuery(DataQuery):

    def __init__(self, sofa_directory_path, hrtf_type='compensated'):
        if hrtf_type == 'raw':
            self._hrtf_type_str = 'R'
        elif hrtf_type == 'compensated':
            self._hrtf_type_str = 'C'
        else:
            raise ValueError(f'Unknown HRTF type "{hrtf_type}"')
        super().__init__('listen', str(Path(sofa_directory_path) / hrtf_type / '44100'), variant_key=hrtf_type)


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[1]) for x in self.sofa_directory_path.glob(f'IRC_????_{self._hrtf_type_str}_44100.sofa')])


class BiLiDataQuery(DataQuery):

    def __init__(self, sofa_directory_path, samplerate=96000, hrtf_type='compensated'):
        if hrtf_type == 'raw':
            self._hrtf_type_str = 'R'
        elif hrtf_type == 'compensated':
            self._hrtf_type_str = 'C'
        elif hrtf_type == 'compensated-interpolated':
            self._hrtf_type_str = 'I'
        else:
            raise ValueError(f'Unknown HRTF type "{hrtf_type}"')
        if samplerate not in (44100, 48000, 96000) or hrtf_type != 'compensated-interpolated':
            samplerate = 96000
        self._samplerate = samplerate
        super().__init__('bili', str(Path(sofa_directory_path) / hrtf_type / str(samplerate)), variant_key=f'{hrtf_type}-{samplerate}')


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[1]) for x in self.sofa_directory_path.glob(f'IRC_????_{self._hrtf_type_str}_HRIR_{self._samplerate}.sofa')])


class ItaDataQuery(DataQuery):

    def __init__(self, sofa_directory_path):
        super().__init__('ita', sofa_directory_path)
        self._default_hrirs_exclude = (2, 14)


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('MRT')[1]) for x in self.sofa_directory_path.glob('MRT??.sofa')])


class HutubsDataQuery(DataQuery):

    def __init__(self, sofa_directory_path, measured_hrtf=True):
        super().__init__('hutubs', sofa_directory_path, variant_key='measured' if measured_hrtf else 'simulated')


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[0].split('pp')[1]) for x in self.sofa_directory_path.glob(f'pp??_HRIRs_{self._variant_key}.sofa')])


class RiecDataQuery(DataQuery):

    def __init__(self, sofa_directory_path):
        super().__init__('riec', sofa_directory_path)


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[3]) for x in self.sofa_directory_path.glob('RIEC_hrir_subject_???.sofa')])


class ChedarDataQuery(DataQuery):

    def __init__(self, sofa_directory_path, radius=1):
        if np.isclose(radius, 0.2):
            self._radius = '02m'
        elif np.isclose(radius, 0.5):
            self._radius = '05m'
        elif np.isclose(radius, 1):
            self._radius = '1m'
        elif np.isclose(radius, 2):
            self._radius = '2m'
        else:
            raise ValueError('The radius needs to be one of 0.2, 0.5, 1 or 2')
        super().__init__('chedar', sofa_directory_path, variant_key=self._radius)


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[1]) for x in self.sofa_directory_path.glob(f'chedar_????_UV{self._radius}.sofa')])


class WidespreadDataQuery(DataQuery):

    def __init__(self, sofa_directory_path, radius=1, grid='UV'):
        if np.isclose(radius, 0.2):
            self._radius = '02m'
        elif np.isclose(radius, 0.5):
            self._radius = '05m'
        elif np.isclose(radius, 1):
            self._radius = '1m'
        elif np.isclose(radius, 2):
            self._radius = '2m'
        else:
            raise ValueError('The radius needs to be one of 0.2, 0.5, 1 or 2')
        if grid not in ('UV', 'ICO'):
            raise ValueError('The grid needs to be either "UV" or "ICO".')
        self._grid = grid
        super().__init__('widespread', sofa_directory_path, variant_key=f'{self._grid}-{self._radius}')


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[1]) for x in self.sofa_directory_path.glob(f'{self._grid}{self._radius}_?????.sofa')])


class Sadie2DataQuery(DataQuery):

    def __init__(self, sofa_directory_path=None, image_directory_path=None, samplerate=96000):
        if samplerate == 44100:
            self._samplerate_str = '44K_16bit_256tap'
        elif samplerate == 48000:
            self._samplerate_str = '48K_24bit_256tap'
        else:
            self._samplerate_str = '96K_24bit_512tap'
        super().__init__('sadie2', sofa_directory_path=sofa_directory_path, image_directory_path=image_directory_path, variant_key=f'{samplerate}')
        self._default_hrirs_exclude = (1, 2, 3, 4, 5, 6, 7, 8, 9) # higher spatial resolution
        self._default_images_exclude = (3, 16) # empty images


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[0][1:]) for x in self.sofa_directory_path.glob(f'[DH]*/[DH]*_HRIR_SOFA/[DH]*_{self._samplerate_str}_FIR_SOFA.sofa')])


    def _all_image_ids(self, side, rear):
        if rear:
            raise ValueError('No rear pictures available in the SADIE II dataset')
        side_str = 'L' if side == 'left' else 'R'
        return sorted([int(x.stem.split('_')[0].split()[0][1:]) for x in self.image_directory_path.glob(f'[DH]*/[DH]*_Scans/[DH]*[_ ]({side_str}).png')])


class ThreeDThreeADataQuery(DataQuery):

    def __init__(self, sofa_directory_path, hrtf_method='measured', hrtf_type='compensated'):
        if hrtf_type == 'raw':
            self._hrtf_type_str = 'BIRs'
        elif hrtf_type == 'compensated':
            self._hrtf_type_str = 'HRIRs'
        elif hrtf_type == 'compensated-lowfreqextended':
            self._hrtf_type_str = 'HRIRs_lfc'
        elif hrtf_type == 'compensated-equalized':
            self._hrtf_type_str = 'HRIRs_dfeq'
        else:
            raise ValueError(f'Unknown HRTF type "{hrtf_type}"')
        if hrtf_method == 'measured':
            self._method_str = 'Acoustic'
        else:
            if hrtf_type not in ('compensated', 'compensated-equalized'):
                raise ValueError('Only compensated and diffuse field equalized types of HRTF available for BEM-simulations')
            if hrtf_method == 'simulated-head':
                self._method_str = 'BEM/Head-Only'
            elif hrtf_method == 'simulated-head_ears':
                self._method_str = 'BEM/Head-and-Ears'
            elif hrtf_method == 'simulated-head_ears_torso-consumer_grade':
                self._method_str = 'BEM/Head-Ears-and-Torso/Consumer-Grade'
            elif hrtf_method == 'simulated-head_ears_torso-reference_grade':
                self._method_str = 'BEM/Head-Ears-and-Torso/Reference-Grade'
            else:
                raise ValueError(f'Unknown HRTF method "{hrtf_method}"')
        super().__init__('3d3a', sofa_directory_path, variant_key=f'{self._method_str}-{hrtf_type}')


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[0].lstrip('Subject')) for x in self.sofa_directory_path.glob(f'{self._method_str}/Subject*/Subject*_{self._hrtf_type_str}.sofa')])


class SonicomDataQuery(DataQuery):

    def __init__(self, sofa_directory_path, samplerate=96000, hrtf_type='compensated'):
        if samplerate not in (44100, 48000, 96000):
            samplerate = 96000
        self._samplerate_str = f'{round(samplerate/1000)}kHz'
        if hrtf_type == 'raw':
            self._hrtf_type_str = 'Raw'
        elif hrtf_type == 'raw-nodelay':
            self._hrtf_type_str = 'Raw_NoITD'
        elif hrtf_type == 'windowed':
            self._hrtf_type_str = 'Windowed'
        elif hrtf_type == 'windowed-nodelay':
            self._hrtf_type_str = 'Windowed_NoITD'
        elif hrtf_type == 'compensated':
            self._hrtf_type_str = 'FreeFieldComp'
        elif hrtf_type == 'compensated-nodelay':
            self._hrtf_type_str = 'FreeFieldComp_NoITD'
        elif hrtf_type == 'compensated-minphase':
            self._hrtf_type_str = 'FreeFieldCompMinPhase'
        elif hrtf_type == 'compensated-minphase-nodelay':
            self._hrtf_type_str = 'FreeFieldCompMinPhase_NoITD'
        else:
            raise ValueError(f'Unknown HRTF type "{hrtf_type}"')
        super().__init__('sonicom', sofa_directory_path, variant_key=f'{hrtf_type}-{samplerate}')


    def _all_hrir_ids(self, side):
        return sorted([int(x.stem.split('_')[0].lstrip('P')) for x in self.sofa_directory_path.glob(f'P????/HRTF/{self._samplerate_str}/P????_{self._hrtf_type_str}_{self._samplerate_str}.sofa')])
