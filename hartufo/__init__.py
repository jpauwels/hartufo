"""A Python toolkit for data-driven HRTF research"""

__version__ = '0.6.6'

from .full import (
    Dataset,
    Cipic,
    Ari,
    Listen,
    CrossMod,
    BiLi,
    Ita,
    Hutubs,
    Riec,
    Sadie2,
    Princeton3D3A,
    Chedar,
    Widespread,
    Scut,
    Sonicom,
    MitKemar,
    CustomSphericalDataset,
)

from .specifications import (
    CollectionSpec,
    SideSpec,
    SubjectSpec,
    HrirSpec,
    HrirPlaneSpec,
    AnthropometrySpec,
    ImageSpec,
    MeshSpec,
)

from .transforms import BatchTransform
