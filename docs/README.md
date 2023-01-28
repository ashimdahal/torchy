# Torchy Docs
Following is the official documentation of torchy wrapper and all the additional functions of the wrapper. This documentation contains detail descriptions of all the arguments that can be passed on the new utilities of nn.Module as well as guides and tutorials to make the best use of torchy wrapper.
## Using model.fit()
|Argument | Default| Description|
|---------|----------|--------|
|train_dataloader| N/A| Pass either the training dataset in form of TensorDataset or the DataLoader for training. Note: It's upon you to pass the validation dataloader if you choose to pass a training dataloader. You don't have to worry about validation dataloader if you passed the TensorDataset.
|loss_fn| N/A| The loss function that should be used to calculate the loss function. Ideal loss function should be from nn.functional but a custom loss function that can handle tensors will work fine.
|opt| N/A| The optimizer that the model should use to fine tune the parameters. Initialize optimizer form torch.optim, set appropirate hyperparameters and pass to .fit().
|epochs|N/A| The number os epochs. Pass integer.
|valid_dataloader| None| Pass a validation DataLoader if you have passed a training DataLoader instead of a Dataset.
|valid_pct| 30 | The percentage of the training TensorDataset that should be the validation dataset. Usual values between 10 and 30.
|batch_size| 32 | The batch size of the Training DataLoader that will be created when you upload a TensorDataset. Ignore if you passed a DataLoader by default.
|accuracy| False| Either to calculate the accuracy of model or not. Pass Boolean True or False.
|device| CPU | The device that the given model, and its dataset should be converted into. Only provide device if you passed a TensorDataset. Value should be any device available ('cpu' or'cuda').

## Using other utility tools on torchy.nn.Module
### model.get_loss()
|Argument | Description|
|-----------|-------------|
|batch| The TensorDataset or batch or tuple containing the input and output tensors in the form of (x,y) whose loss is to be calculated|
|loss_fn| The loss function from nn.functional should be passed.|
### model._accuracy()
|Argument | Description|
|-----------|-------------|
|labels| The actual outputs.|
|preds|The predicted outputs.|

### model.validate()
|Argument | Description|
|-----------|------------|
|valid_dl | The validation dataloader which should be used to do validation|
|loss_fn| The loss function from nn.functional should be passed.|

PS: When using model.validate() you don't need to do model.eval() beforehand as the method is already decorated with `@torch.no_grad()` and has `self.eval()` implemented whithin the method.

## What additional utilities are in torchy.utils.data?

###  DeviceDL
DeviceDL is a helper tool to put your DataLoader into given device in the most efficient way. It's not recommended to put the entire DataLoader into given device in pytorch, so DeviceDL would only put the current batch in the specified device and pytorch will automatically remove the batch from device after its processed.

DeviceDL can be used in the following way:
```python
from torchy.utils.data import DeviceDL, DataLoader, TensorDataset
# create a TensorDataset
dataset = TensorDataset(x,y)
# create your desired dataloader with the hyperparameters
dataloader = DataLoader(dataset, ...)
# now put your dataloader into the appropirate device using DeviceDL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataloader_device = DeviceDL(dataloader, device)
```
|Argument | Description|
|-----------|------------|
|dl| The dataloader that you have created using for your model.|
|dev| The device that the given dataloader should be kept in.|


### SplitPCT
Do you find it a `meh` experience when you have to count the number of examples you want to use in your training and validation set when using random_split? you don't want to install sklearn in your virtual environment just to split your dataset into train and validation sets?

If so, SplitPCT is a native pytorch's implementation to split the given dataset into training and validaiton set based on the percentage of data you want in training set and not the number of examples that the dataset should be divided into.

SplitPCT can be used in the following way:
```python
from torchy.utils.data import SplitPCT, TensorDataset
# create a dataset
dataset = TensorDataset(x,y)
# determine what percentage of data should be on your training dataset
training_pct = 75
dataset = SplitPCT(dataset,training_pct)
# get training dataset and validation dataset as attributes
train_ds, validaiton_ds = dataset.train_ds, dataset.validation_ds
# get the original TensorDataset
orig_dataset = dataset.tensor_dataset
```
PS: the dataset passed to SplitPCT can be any type of pytorch's dataset and should not be limited to TensorDataset.
|Argument | Description|
|-----------|------------|
|tensor_dataset| The dataset that you have created using for your model.|
|train_pct| The percentage of data that sould be on the training dataset; rest will be validation dataset.|
## Effect on torch.nn

By definition, torchy is a pytorch wrapper, so there will be no changes on torch.nn or any other torch functionality. torchy.nn can replacetorch.nn there will not be any unsolvable errors.
# Torchy user guides and tutorials
Examples and quick start guide to use torchy can also be found on the project readme at [github](https://github.com/ashimdahal/torchy#additional-functionality). 

Since torchy is just a Wrapper and doesn't implement everything from scratch its recommended to implement just the nn.Module using the wrapper. 

Recommended
```python
import torch
import torchy.nn as nn
import torch.nn.functional as F
```
Make your models as you would using torch.nn
```py
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self,x):
        return self.linear(x)


model = Model()
```
Choose your loss function and optimizers
```py
loss_fn = F.mse_loss
opt = torch.optim.SGD(model.parameters(), lr=0.001)
```
Then, you can use torchy's DeviceDL to put your DataLoader in the given device

```py
from torchy.utils.data import DeviceDL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataloader = DeviceDL(old_dataloader, device)
valid_dataloader = DeviceDL(old_valid_dataloader, device)
```

Now you can fit the model
```py
hist = model.fit(
    train_dataloader,
    loss_fn, 
    opt, 
    epochs, 
    valid_dataloader, 
    batch_size=64, 
    accuracy=True, 
    device=device
)
```
If you don't want to go through the hassle of making a dataloader then don't worry, torchy will do it for you. 

PS: Torchy requires a TensorDataset to be passed for the following implementation to work.

```py
hist = model.fit(
    dataset,
    loss_fn,
    opt,
    epochs,
    valic_pct=30,
    batch_size=64,
    accuracy=True,
    device=device
)
```
Looks pretty same and simple with the only change being you have to provide the percentage of the dataset that should be in the validation dataset and eventually the validation dataloader.
# Why torchy is better than other pytorch .fit() implementations
Because the wheel doesn't need to be reinvented when using torchy. The end user can just use torchy as torch and just learn some new methods in the nn.Module that are handy. 