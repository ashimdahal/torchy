class TorchNotInstalledError(Exception):
    """
    Exception raised when torch is not installed beforehand.
    This Exception shouldn't occur if easytorch is installed via pip. 
    If any developer finds himself editing the source code of easytorch and has a different virtual environment running, 
    TorchNotInstalledError will be handy in that usecase.

    Attributes:
        message -- explanation of the error
    """

    def __init__(
        self,
        message = (
        "Torch is not installed properly in your device." 
        "Please make sure that you have pyTroch installed to run this module."
        "You can use 'pip3 install torch' to install torch if it wan't automatically installed in "
        "your system during installation of easytorch module. "
        "Be sure to check your virtual environments and make sure that you have torch installed "
        "in the one you are currently using. View full guide to install pytorch in your "
        "local machine: https://pytorch.org/get-started/locally/ "
        )
        ):
        self.message = message
        super().__init__(self.message)
        

try:
    import torch.nn as torch_nn
    import torch
    from tqdm import tqdm
except ModuleNotFoundError:
    raise TorchNotInstalledError()


class Module(torch_nn.Module):
    def get_loss(self, batch, loss_fn):
        ...
    
    def __valid_step():
        ...
    
    def __mean_validation():
        ...
    
    def __log_epoch():
        ...

    def fit(self, dataloader, loss_fn, opt, epochs, valid = None,) -> tuple[dict,int]:
        self.hist = []
        for e in tqdm(range(epochs),desc="Running Epoch"):

            self.train()
            self.train_loss =[]
            for batch in tqdm(dataloader,desc='Running Batch'):

                loss = self.get_loss(batch,loss_fn)
                self.train_loss.append(loss)
                loss.backward()
                opt.step()
                opt.zero_grad()

    @classmethod
    def from_model():
        ...
    