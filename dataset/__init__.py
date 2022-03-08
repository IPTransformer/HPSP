from .cihp import CIHPDataset
from .lip import LIPDataset
from .mhp import MHPDataset

__all__ = ['CIHPDataset', 'LIPDataset', 'MHPDataset']

def build_dataset(type):
    if type == 'cihp':
        return CIHPDataset
    elif type == 'lip':
        return LIPDataset
    elif type == 'mhp':
        return MHPDataset