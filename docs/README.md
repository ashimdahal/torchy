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
torchy.utils.data has 2 new additional utilities:

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

### SplitPCT

## Effect on torch.nn

# Torchy user guides and tutorials

# Why torchy is better than other pytorch .fit() implementations