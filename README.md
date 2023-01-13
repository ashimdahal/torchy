# Welcome to torchy
The aim of this project is to create a PyTorch wrapper that wraps the torch.nn.Module and has additional data preprocessing utilities on torch.utils.data.
We aim to retain every functionality of PyTorch, while keeping them native, and also add our piece of functionality.

<b>The aim of torchy is to enhance the experience of Pytorch and not to replace it. torchy is ready to be used in everyday code and is in a beta stage as of today. After additional checks and testing, torchy will be passed as stable.</b>
## Introduction
torchy is a PyTorch wrapper that has some additional benefits to using plain pytorch. With torchy you have everything in pytorch plus
some additional features found on other libraries. The main separating factor between torchy and torchfit or 100s of other pytorch-like
modules that exists is that you don't have to re-learn to use the pytorch module.

torchy is a wrapper build on top of pytorch which enables you to use your existing code on pyTorch and still have the added benefits.
## Installation using pip
It's a good idea to have PyTroch preinstalled on your current virtual environment. See [official guide](https://pytorch.org/get-started/locally/) to install PyTorch. 

<i>It's recommended to have python version >=3.6 and <=3.8, although no problems have yet been encountered in 3.9, and 3.10.</i>

Use pypi's pip to install torchy.
``` 
pip install torchy 
```
or
```
pip3 install torchy
```

PS: PyTorch will be atuomatically installed to your environment if you already don't have it but it's recommended to install it using the official guide.
## Additional Functionality
Define a model using nn.Module just like with regular pyTorch but `import torchy.nn` instead of `torch.nn`.
```py
import torchy.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self,x):
        return self.linear(x)


model = Model()
```
Now you can use torchy's functionality.

```py
import torch
from torchy.utils.data import TensorDataset, DataLoader

# prepare dummy data
x = torch.tensor([[12.],[13],[15]])
y = torch.tensor([[2.],[3],[4]])
dataset = TensorDataset(x,y)

# nn is still same (torchy.nn)
loss_fn = nn.functional.mse_loss
opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=.9)
# Use mode.fit() to fit the model in the given TensorDataset
model = model.fit(dataset, loss_fn, opt, epochs=20, valid_pct=25, batch_size=2)
# Now you have a trained model and can also access model.hist attribute
print(model.hist)
```
You can also use a dataloader instead of a dataset. 
```python
# Use a DataLoader instead of a TensorDataSet
dataloader = DataLoader(dataset, batch_size = 2)
model = model.fit(dataloader, loss_fn,opt,20)
```
If you're using a dataloader and want to do validation while running .fit()
after every epochs, you will have to manually pass valid_dataloader.


`torchy.utils.data` can also be used to put your dataloader into a device and split your dataset.
```py
from torchy.utils.data import DeviceDL, SplitPCT
# put dataloader in appropirate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataloader = DeviceDL(dataloader)

# Split the dataset
dataset = SplitPCT(dataset)
train_ds, valid_ds = dataset.train_ds, dataset.valid_ds
```

Additional features like get_loss(), _accuracy() and full documentation, user guide, best practices and tutorials to use torchy can be found in the [docs](https://github.com/ashimdahal/torchy/blob/master/docs/README.md).

## To-do
0. More testing

Feel free to contribute your code and drop a star on the project if you liked the idea.