from utils.experiment import *
from utils.visualization import *
from utils.metrics import D1_metric, Thres_metric, EPE_metric

from utils.losses import *


__loss_types__ = {
    "mf_main": CriterionPairWiseforWholeFeatAfterPool(scale=0.5),
    "mf_4D": CriterionPairWiseforWholeFeatAfterPool4D(scale=0.1),
    "mf_old": CriterionFeaSum(),
    "hint": HintLoss(),
    "hint_smooth": HintLossSmooth(),
    "hint_huber": HintLossHuber(),
    "rkd": RKDLoss(),
    "soft_kl": CriterionPixelWise(),
    "soft_kl4D": CriterionPixelWise4D(),
    "soft_kl4D_v2": CriterionPixelWise4D_v2(),
    "Focal": FocalLoss(),
    "StereoKLD": StereoKLD(),
    "StereoSimilarity": StereoSimilarity(),
    "LogL1": LogL1Loss(),
    "LogL1_v2": LogL1Loss_v2(),
    "SmoothL1": nn.SmoothL1Loss(size_average=True)
}