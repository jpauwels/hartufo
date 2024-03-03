"""A Python toolkit for data-driven HRTF research"""

__version__ = '0.6.4'

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

from .planar import (
    CipicPlane,
    AriPlane,
    ListenPlane,
    CrossModPlane,
    BiLiPlane,
    ItaPlane,
    HutubsPlane,
    RiecPlane,
    Sadie2Plane,
    Princeton3D3APlane,
    ChedarPlane,
    WidespreadPlane,
    ScutPlane,
    SonicomPlane,
    MitKemarPlane,
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
