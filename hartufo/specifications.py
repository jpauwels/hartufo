from abc import ABC
from typing import Callable, Iterable, Optional, Union


class Spec(ABC):

    name = ''


    def __init__(self,
        preprocess: Optional[Union[Callable, Iterable[Callable]]] = None,
        transform: Optional[Union[Callable, Iterable[Callable]]] = None,
    ):
        self.spec = {}
        self.spec['preprocess'] = sanitise_callables(preprocess)
        self.spec['transform'] = sanitise_callables(transform)


    def add(self, key, value):
        if value is not None:
            self.spec[key] = value


def sanitise_callables(x: Optional[Union[Callable, Iterable[Callable]]]) -> Iterable[Callable]:
    if x is None:
        return []
    if callable(x):
        return [x]
    return x


def sanitise_specs(x: Optional[Union[Spec, Iterable[Spec]]]) -> Iterable[Spec]:
    if x is None:
        return ()
    if isinstance(x, Spec):
        return (x,)
    return x


def sanitise_multiple_specs(*multi_specs: Iterable[Optional[Union[Spec, Iterable[Spec]]]]) -> Iterable[Spec]:
    return tuple((spec for specs in multi_specs for spec in sanitise_specs(specs)))


class CollectionSpec(Spec):

    name = 'collection'


    def __init__(self,
        preprocess: Optional[Union[Callable, Iterable[Callable]]] = None,
        transform: Optional[Union[Callable, Iterable[Callable]]] = None,
    ):
        super().__init__(preprocess, transform)


class SideSpec(Spec):

    name = 'side'


    def __init__(self,
        preprocess: Optional[Union[Callable, Iterable[Callable]]] = None,
        transform: Optional[Union[Callable, Iterable[Callable]]] = None,
    ):
        super().__init__(preprocess, transform)


class SubjectSpec(Spec):

    name = 'subject'


    def __init__(self,
        preprocess: Optional[Union[Callable, Iterable[Callable]]] = None,
        transform: Optional[Union[Callable, Iterable[Callable]]] = None,
    ):
        super().__init__(preprocess, transform)


class HrirSpec(Spec):

    name = 'hrir'


    def __init__(self,
        domain: str = 'time',
        side: Optional[str] = None,
        fundamental_angles: Optional[Iterable[float]] = None,
        orthogonal_angles: Optional[Iterable[float]] = None,
        distance: Optional[Union[float, str]] = None,
        additive_scale_factor: Optional[float] = None,
        multiplicative_scale_factor: Optional[float] = None,
        samplerate: Optional[float] = None,
        length: Optional[float] = None,
        min_phase: bool = False,
        exclude: Optional[Iterable[int]] = None,
        preprocess: Optional[Union[Callable, Iterable[Callable]]] = None,
        transform: Optional[Union[Callable, Iterable[Callable]]] = None,
    ):
        super().__init__(preprocess, transform)
        self.add('domain', domain)
        self.add('side', side)
        self.add('fundamental_angles', fundamental_angles)
        self.add('orthogonal_angles', orthogonal_angles)
        self.add('distance', distance)
        self.add('additive_scale_factor', additive_scale_factor)
        self.add('multiplicative_scale_factor', multiplicative_scale_factor)
        self.add('samplerate', samplerate)
        self.add('length', length)
        self.add('min_phase', min_phase)
        self.add('exclude', exclude)


class HrirPlaneSpec(HrirSpec):

    def __init__(self,
        plane,
        domain: str = 'time',
        side: Optional[str] = None,
        plane_angles: Optional[Iterable[float]] = None,
        plane_offset: float = 0.,
        positive_angles: bool = False,
        distance: Optional[Union[float, str]] = None,
        additive_scale_factor: Optional[float] = None,
        multiplicative_scale_factor: Optional[float] = None,
        samplerate: Optional[float] = None,
        length: Optional[float] = None,
        min_phase: bool = False,
        exclude: Optional[Iterable[int]] = None,
        preprocess: Optional[Union[Callable, Iterable[Callable]]] = None,
        transform: Optional[Union[Callable, Iterable[Callable]]] = None,
    ):
        if plane not in ('horizontal', 'median', 'frontal', 'vertical', 'interaural'):
            raise ValueError('Unknown plane "{}", needs to be "horizontal", "median", "frontal", "vertical" or "interaural".')
        super().__init__(domain, side, None, None, distance, additive_scale_factor, multiplicative_scale_factor, samplerate, length, min_phase, exclude, preprocess, transform)
        self.add('plane', plane)
        self.add('plane_angles', plane_angles)
        self.add('plane_offset', plane_offset)
        self.add('positive_angles', positive_angles)


class AnthropometrySpec(Spec):

    name = 'anthropometry'


    def __init__(self,
        side: Optional[str] = None,
        select: Optional[str] = None,
        partial: bool = False,
        preprocess: Optional[Union[Callable, Iterable[Callable]]] = None,
        transform: Optional[Union[Callable, Iterable[Callable]]] = None,
    ):
        super().__init__(preprocess, transform)
        self.add('side', side)
        self.add('select', select)
        self.add('partial', partial)


class ImageSpec(Spec):

    name = 'image'


    def __init__(self,
        side: Optional[str] = None,
        rear: bool = False,
        preprocess: Optional[Union[Callable, Iterable[Callable]]] = None,
        transform: Optional[Union[Callable, Iterable[Callable]]] = None,
    ):
        super().__init__(preprocess, transform)
        self.add('side', side)
        self.add('rear', rear)


class MeshSpec(Spec):

    name = '3d-model'


    def __init__(self,
        preprocess: Optional[Union[Callable, Iterable[Callable]]] = None,
        transform: Optional[Union[Callable, Iterable[Callable]]] = None,
    ):
        super().__init__(preprocess, transform)
