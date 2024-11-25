from .kitti_dataset import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .drivingstereo_dataset import DrivingStereoDataset
from .middlebury_dataset import MiddleBuryDataset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "drivingstereo": DrivingStereoDataset,
    "middlebury": MiddleBuryDataset
}
