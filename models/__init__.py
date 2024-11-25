from utils import losses

#teachers
from models.gwcnet_xsfe_2hg_slied8 import GwcNet_G
from models.gwc_lac.stackhourglass import PSMNet as Gwcnet_LAC
from models.gwc_acvn.acv import ACV as ACVNet

#students
from models.gwcnet_xsfe_2hg_slied16 import GwcNet_G as gwcnet_xsfe_2hg_slied16
from models.gwcnet_xsfe_2hg_slied16_wo import GwcNet_G as gwcnet_xsfe_2hg_slied16_wo

from models.gwcnet_xsfe_2hg_slied16_amod import GwcNet_G as gwcnet_xsfe_2hg_slied16_amod

__models__ = {
    "gwcnet-g": GwcNet_G,
    "gwcnet-lac": Gwcnet_LAC,
    "acvnet": ACVNet,
    "gwcnet-xsfe-2hg-slied16": gwcnet_xsfe_2hg_slied16,
    "gwcnet-xsfe-2hg-slied16-wo": gwcnet_xsfe_2hg_slied16_wo,
    "gwcnet-xsfe-2hg-slied16-amod": gwcnet_xsfe_2hg_slied16_amod
}

