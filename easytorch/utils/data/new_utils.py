from ...nn import TorchNotInstalledError

try:
    from torch.utils.data import random_split
except ModuleNotFoundError:
    raise TorchNotInstalledError()


class DeviceDL:
    
    def __init__(self,dl,dev):
        self.dl = dl
        self.dev = dev
    
    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch,self.dev)
            
    def __len__(self):
        return len(self.dl)
