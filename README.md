# Welcome to torchy
torchy is a work in progress and will be going through constant changes everyday.
## Introduction
easy-torch is a PyTorch wrapper that has some additional benefits to using plain pytorch. With easy-torch you have everything in pytorch plus
some additional features found on other libraries. 
## Installation using pip
## Additional Functionality
```python
import torchy.nn as nn
import torch
from torchy.utils.data import TensorDataset, DataLoader, random_split, DeviceDL

x = torch.tensor([[12.],[13],[15]])
y = torch.tensor([[2.],[3],[4]])
train = TensorDataset(x,y)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self,x):
        return self.linear(x)

loss_fn = nn.functional.mse_loss
model = Model()
opt = torch.optim.SGD(model.parameters(),lr=0.001,momentum=.9)
model = model.fit(train, loss_fn,opt,20,valid_pct = 20,batch_size=2)
```
You can also use a dataloader instead of a dataset. If you're using a dataloader be sure to pass additional argument "valid_dataloader" otherwise 
the no model validation would be carried out.

```python
dl = DataLoader(train,batch_size = 2)
model = model.fit(dl, loss_fn,opt,20)

```

## To-do
more documentation and all arguments and their function table comming soon.