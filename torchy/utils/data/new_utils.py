from torch.utils.data import random_split


class DeviceDL:
    '''Puts the given dataloader into specified device
    Pytorch still doesn't have a way to put a dataloader into a device.
    '''
    
    def __init__(self, dl, dev):
        '''Initializes the Device DataLoader
        Arguments:
            dl -- the dataloader to put in the given device.
            dev -- the device to put the dataloader into.
        '''

        self.dl = dl
        self.dev = dev
    
    def to_device(self, data, device):
        '''Helper method to put the given data into device
        Arguments:
            data -- the data to put into given device
            device -- the device to put the data into
        Returns:
            The given data(either a model or a batch of data from a dataloader)
            converted into the given device
        '''

        if isinstance(data,(list,tuple)):
            return [self.to_device(d,device) for d in data]
        return data.to(device, non_blocking = True)
    
    def __iter__(self):
        '''Dunder method to yeild the batch into device.
        '''

        for batch in self.dl:
            yield self.to_device(batch, self.dev)
            
    def __len__(self):
        '''Dunder method to return the length of the DataLoader
        '''

        return len(self.dl)


class SplitPct:
    def __init__(self, tensor_dataset, train_pct):
        '''Initializes tensor_dataset and train_pct and
        adds two new attributes to the instance:
            self.train_ds = the training dataset after random_split
            self.valid_ds = the validation dataset after random_split
        
        Arguments:
            tensor_dataset -- the TensorDataset to split into train and validation dataset
            train_pct -- the percentage of given samples that sould be in the training dataset
        '''

        self.train_pct = train_pct
        self.tensor_dataset = tensor_dataset

        train_num, valid_num  = self.pct_to_val()
        self.train_ds, self.valid_ds = random_split(self.tensor_dataset, [train_num, valid_num])

    def pct_to_val(self):
    
        '''Helper function to make code cleaer.
        changes percentage split into numbers of data.
        INPUTS:
        train_pct: the percentage of training data 
        valid_pct: the percentage of validation data
        data: the dataset
        returns: numbers of data
        '''

        train_num = int(self.train_pct/100*len(self.tensor_dataset))
        valid_num = int(len(self.tensor_dataset) - self.train_num)
        return train_num , valid_num
        