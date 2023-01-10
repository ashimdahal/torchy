from ...nn import TorchNotInstalledError

try:
    from torch.utils.data import *
except ModuleNotFoundError:
    raise TorchNotInstalledError()

from .new_utils import *