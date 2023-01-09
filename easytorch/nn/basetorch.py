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
        self.loss_fn = loss_fn
        x, y = batch
        y_hat = self(x)
        loss =  self.loss_fn(y_hat, y)

        return loss
    
    def __valid_step(self, batch, loss_fn):
        loss = self.get_loss(batch, loss_fn)
        if self.accuracy:
            x,y = batch
            y_hat = self(x)
            acc = self._accuracy(labels = y_hat, preds = y)
            return {'valid_loss' : loss , 'valid_acc' : acc}
        return {'valid_loss': loss}

    @staticmethod
    def _accuracy(labels,preds):
        acc = torch.sum(torch.round(preds) == labels) / len(labels)
        return acc

    @staticmethod
    def __mean_validation(out):
        loss = torch.stack([l['valid_loss'] for l in out]).mean()
        if self.accuracy:
            acc = torch.stack([l['valid_acc'] for l in out]).mean()
    
    def __log_epoch(self,e,res):
        ...

    @torch.no_grad()
    def validate(self, valid_dl, loss_fn):
        self.eval()
        out = [self.__valid_step(batch, loss_fn) for batch in valid_dl]
        return self.__mean_validation(out)

    def fit(
        self,
        train_dataloader, 
        loss_fn, 
        opt, 
        epochs, 
        valid_dataloader = None, 
        valid = 30, 
        accuracy = False
        ):
        self.accuracy = accuracy 
        self.hist = []
        for e in tqdm(range(epochs),desc="Running Epoch"):

            self.train()
            self.train_loss =[]
            for batch in tqdm(train_dataloader, desc='Running Batch'):

                loss = self.get_loss(batch, loss_fn)
                self.train_loss.append(loss)
                loss.backward()
                opt.step()
                opt.zero_grad()
            
            res = self.validate(valid_dataloader, loss_fn)
            res['train_loss'] = torch.stack(train_loss).mean().item()
            
            self.__log_epoch(e,res)
            self.hist.append(res)
        return self, self.hist

    @classmethod
    def from_model():
        ...
    