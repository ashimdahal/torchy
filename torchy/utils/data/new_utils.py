from torch.utils.data import random_split

class DeviceDL:

    def to_device(self, data, device):
        if isinstance(data,(list,tuple)):
            return [self.to_device(d,device) for d in data]
        return data.to(device,non_blocking = True)
    
    def __init__(self,dl,dev):
        self.dl = dl
        self.dev = dev
    
    def __iter__(self):
        for batch in self.dl:
            yield self.to_device(batch,self.dev)
            
    def __len__(self):
        return len(self.dl)


class SplitPct:
    def __init__(self, tensor_dataset, train_pct):
        self.train_pct = train_pct
        self.tensor_dataset = tensor_dataset

        train_num, valid_num  = self.pct_to_val()
        self.train_ds, self.valid_ds = random_split(dataset, [train_num, valid_num])

    def pct_to_val(self):
    
        '''Helper function to make code cleaer.
        changes percentage split into numbers of data.
        INPUTS:
        train_pct: the percentage of training data 
        valid_pct: the percentage of validation data
        data: the dataset
        returns: numbers of data'''

        train_num = int(self.train_pct/100*len(self.tensor_dataset))
        valid_num = int(len(self.tensor_dataset) - self.train_num)
        return train_num , valid_num
        