import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

'''
transform: callable function to transform the data. 
transforms: python module hosting many key transformations. 
.Compose: class, lets us queue up many transformations in a row. 
.ToTensor: reorders size (HWC to CHW), which PyTorch expects. Also converts PIL image / numpy ndarray (in this case, PIL image) into FloatTensor. Also normalizes values to [0,1]
'''
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(
    root = './data', # save it to ./data if not there, otherwise get it from there. (default: no default exists)
    train = True, # get the training split of MNIST (60,000 images) (default: True)
    download = True, # downloads it if it is not already present (default: False)
    transform = transform, # applies the transform described above (default: None)
    target_transform = None # applies a transform to the target data (labels). (default: None)
)

test_dataset = datasets.MNIST(
    root = './data',
    train = False, # we want the test data, this is the one difference.
    download = True,
    transform = transform,
    target_transform = None
)

train_loader = DataLoader(
    dataset = train_dataset, # tells it what dataset to wrap (default: No default, required)
    batch_size = 64, # the size of batches used (default: 1)
    shuffle = True, # shuffle the data to prevent learning the order (default: False)
    num_workers = 0, # I'm on a CPU (same as default value)
    drop_last = False, # Keep the last smaller batch if it exists, no need to chop it off to guarantee uniform sizes. (ex, no need for batch norm)
    pin_memory = False # I'm on a CPU not GPU, so this isn't helpful (same as default)
)

test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = 64,
    shuffle = True,
    num_workers = 0,
    drop_last = False,
    pin_memory = False
)
