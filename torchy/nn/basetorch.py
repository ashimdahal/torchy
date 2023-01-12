class TorchNotInstalledError(Exception):
    """Exception raised when torch is not installed beforehand.
    This Exception shouldn't occur if torchy is installed via pip. 
    If any developer finds himself editing the source code of torchy and has a different virtual environment running, 
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
        "your system during installation of torchy module. "
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
    from ..utils.data import DeviceDL, random_split, DataLoader
except ModuleNotFoundError:
    raise TorchNotInstalledError()


class Module(torch_nn.Module):
    """Overrides while inheriting pytorch's nn.Module.
    This version of Module is to introduce easy-torch's functionality of .fit() on
    any models cerated by inheriting easy-torch.nn.Module class.
    """

    def get_loss(self, batch, loss_fn):
        """Calculates loss of the given batch.

        Attributes:
            batch -- Batch from either the dataloader or the dataset overall.
            loss_fn -- any sort of loss function from torchy.nn.functional. 
        Returns:
            Returns the loss calculated by the loss function

        Call it: model.get_loss(batch,loss_fn)
        """

        self.loss_fn = loss_fn
        x, y = batch
        y_hat = self(x)
        loss =  self.loss_fn(y_hat, y)

        return loss
    
    def __valid_step(self, batch, loss_fn):
        """Validates the given batch with loss function
        """

        loss = self.get_loss(batch, loss_fn)
        if self.accuracy:
            x,y = batch
            y_hat = self(x)
            acc = self._accuracy(labels = y_hat, preds = y)
            return {'valid_loss' : loss , 'valid_acc' : acc}
        return {'valid_loss': loss}

    @staticmethod
    def _accuracy(labels, preds):
        """Calculates accuracy for a generic classification tasks.

        Attributes:
            labels: the actual y value/output of the model
            preds: the predicted y value/preductions for the model
        Returns:
            Accuracy of the model based on mathematical formula 
                True Positives + True Negatives / (True Positives + True Negatives + False Positives + False Negatives)

        Call it: model._accuracy(y,y_hat)
        """

        acc = torch.sum(torch.round(preds) == labels) / len(labels)
        return acc

    def __mean_validation(self, out):
        """Internal function to calculate the mean loss and accuracy of validation step
        """

        loss = torch.stack([l['valid_loss'] for l in out]).mean()
        if self.accuracy:
            acc = torch.stack([l['valid_acc'] for l in out]).mean()
            return {'valid_loss': loss.item() , 'valid_acc': acc.item()}

        return {'valid_loss': loss.item(), }
    
    def __log_epoch(self,e,res):
        """ Prints details about current epoch based on the current model's properties.
        """

        if self.valid_dataloader is None:
            print((
                "[{} / {}] epoch/s,"
                "training loss is {:.4f} No Validation Loss and Accuracy Because you didn't provide validation dataloader"
                )
            .format(
                e+1,self.epochs,
                res['train_loss'],
                ))
            return 0

        if self.accuracy:
            print((
                "[{} / {}] epoch/s,"
                "training loss is {:.4f} validation loss is {:.4f}, validation accuracy is {:.4f} "
                )
            .format(
                e+1,self.epochs,
                res['train_loss'],
                res['valid_loss'],
                res['valid_acc']
                ))
        else:
            print((
                "[{} / {}] epoch/s,"
                "training loss is {:.4f} validation loss is {:.4f} "
                )
            .format(
                e+1,self.epochs,
                res['train_loss'],
                res['valid_loss'],
                ))
    @staticmethod
    def pct_to_val(train_pct,data):
        '''Helper function to make code cleaer.
        changes percentage split into numbers of data.

        Attributes:
            train_pct -- the percentage of training data 
            data -- the dataset
        Returns: 
            number of data that sould be on training and validation sets
        '''

        train_num = int(train_pct/100*len(data))
        valid_num = int(len(data) - train_num)
        return train_num , valid_num

    def to_device(self, data, device):
        ''' puts the given data into specified device
        '''
        if isinstance(data,(list,tuple)):
            return [self.to_device(d,device) for d in data]
        return data.to(device,non_blocking = True)

    @torch.no_grad()
    def validate(self, valid_dl, loss_fn):
        self.eval()
        out = [self.__valid_step(batch, loss_fn) for batch in valid_dl]
        return self.__mean_validation(out)

    def _fit_dataloader(self):
        """Fits self/model on the given dataloader
        
        Attributes:
            self -- the model and its different attributes. See fit() 
                    for details of the attributes written here
        Returns:
            trained self/model
        """
        self.hist = []
        for e in range(self.epochs):

            self.train()
            self.train_loss =[]
            for batch in tqdm(self.train_dataloader, desc='Running Batch'):
                loss = self.get_loss(batch, self.loss_fn)
                self.train_loss.append(loss)
                # Back Propagation
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

            if self.valid_dataloader is None:
                res = {
                    'train_loss':torch.stack(self.train_loss).mean().item()
                    }
                self.hist.append(res)
                self.__log_epoch(e, res)
            else:
                res = self.validate(self.valid_dataloader, self.loss_fn)
                res['train_loss'] = torch.stack(self.train_loss).mean().item()

                self.__log_epoch(e,res)
                self.hist.append(res)
        return self

    def _fit_dataset(self):
        """Fits self/model on the given TensorDataset.
        The model currently can only accept TensorDataset as a dataset to train 
        the model.
        
        Attributes:
            self -- the model and its different attributes. See fit() 
                    for details of the attributes written here
        Returns:
            trained self/model
        """
        train_ds, valid_ds = random_split(self.tensor_ds, self.pct_to_val(self.train_pct, self.tensor_ds))

        train_dl = DeviceDL(DataLoader(train_ds, batch_size = self.batch_size), self.device)
        valid_dl = DeviceDL(DataLoader(valid_ds, batch_size = self.batch_size,), self.device)
        self = self.to_device(self,self.device)

        self.valid_dataloader = valid_dl
        self.hist = []
        for e in range(self.epochs):
            self.train()
            self.train_loss = []
            for batch in tqdm(train_dl, desc="Running Batch"):
                loss = self.get_loss(batch,self.loss_fn)
                self.train_loss.append(loss)

                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

            res = self.validate(valid_dl, self.loss_fn)
            res['train_loss'] = torch.stack(self.train_loss).mean().item()

            self.__log_epoch(e, res)
            self.hist.append(res)
        return self

    def fit(self,
            train_dataloader, 
            loss_fn, 
            opt, 
            epochs, 
            valid_dataloader=None, 
            valid_pct=30, 
            batch_size=32,
            accuracy=False,
            device='cpu'
        ):
        '''Fits the given dataloader or dataset to the model/self.
        If dataloader is given, model cannot generate its own validation dataloader so
        it's a good practice to pass valid_dataloader while passing a train dataloader.
        
        If dataset is provided dataloader for both train and validation sets is created
        and the model is fitted based on the given parameters.

        Attributes:
            train_dataloader -- either the training dataloader or the entire dataset to fit the model into
            loss_fn -- an appropirate function choosed from nn.functional
            opt -- an appropirate function choosed from torch.optim
            epochs -- number of epochs to train the model
            valid_dataloader -- the validation dataloader, only needed if training dataloader is provided 
                instead of a TensorDataset.
            valid_pct -- the percentage of validation batch from the TensorDataset. 
            batch_size -- only needed if TensorDataset is provided-> the batch size for each training dataloader
                and validation dataloader
            accuracy -- True or False, a boolean to either calculate generic accuracy of the model or not
            device -- The device to use to put the dataloader or dataset into.

        Returns:
            trained model based on _fit_dataset() or _fit_dataloader
        '''
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.opt = opt
        self.valid_dataloader = valid_dataloader
        self.accuracy = accuracy 
        self.batch_size=batch_size

        if 'DataLoader' in str(type(train_dataloader)):
            self.train_dataloader = train_dataloader
            return self._fit_dataloader()

        if 'TensorDataset' in str(type(train_dataloader)):
            self.tensor_ds = train_dataloader
            self.train_pct = 100 - valid_pct
            self.device = device
            return self._fit_dataset()

        print(
            ("Please either provide a DataLoader or a TensorDataset"
            " to train the model into. Read the docs: https://github.com/ashimdahal/easy-torch"))