
from .hpsp_resnet import main_model as res_cihp
from .hpsp_swin import HPSP

__all__ = ['res_cihp', 'HPSP']


def build_cihp_model(type):
    if type.startswith('res'):
        return res_cihp
    elif type.startswith('Swin'):
        return HPSP
        # return IPTR
    # elif type.startswith('nSwin'):
    #     return new_iptr
