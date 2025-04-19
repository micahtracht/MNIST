import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from CNN import SimpleCNN

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
    shuffle = False, # false so that we get the test data in a predictable order - won't affect performance or overfitting.
    num_workers = 0,
    drop_last = False,
    pin_memory = False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
# move the model to gpu if available, otherwise to the cpu.
model = model.to(device)

model = SimpleCNN() # instantiate the model. It now has everything we gave it in __init__ of SimpleCNN, which includes the parent nn.Module class.
criterion = nn.CrossEntropyLoss() # from torch.nn, measures -log(softmax likelihoods), or the distance between our answer and the true, one-hot one. Think of it like how surprised the model was.
optimizer = optim.Adam(model.parameters(), lr = 0.001) # this uses the Adaptive Moment Estimation to compute gradients, and based on that, optimize the model. It does this scaled by a learning rate (lr=0.001), and keeps track of average gradients (exponentially decayed) to make its estimates.

epochs = 10 # this is the number of times we will train over all of the data

# training loop
for epoch in range(epochs):
    # this means that the model will exhibit dropout and batchnorm, which are not present during evaluation.
    model.train() # this sets the model to training mode - not evaluation mode.
    
    running_loss = 0.0 # this measures the total amount of loss over all batches in the epoch.
    # this lets us display the average loss per epoch of the model.
    
    # enumerate the dataloader object, train_loader. Iterate over the batch_idx, and a tuple containing (inputs, labels) of it, which are each lists.
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # move the inputs and labels to the same device (cpu or gpu) as the model.
        inputs, labels = inputs.to(device), labels.to(device)
        
        # autograd does not automatically clear gradients to 0 (accumulation is sometimes useful), so this code does that
        optimizer.zero_grad()
        
        # get outputs from the model (gets a batch). This implicitly calls forward().
        outputs = model(inputs)
        
        # compute the cross-entropy loss between our outputs and the labels
        loss = criterion(outputs, labels)
        
        # performs backprop (goes in reverse order using chainrules to update the weights to minimize loss)
        loss.backward()
        
        # actually update the parameters using the optimizer
        optimizer.step()
        
        running_loss += loss.item()
    
    average_loss = running_loss / len(train_loader)
    print(f'Epoch # {epoch+1} out of {epochs} epochs. Average loss: {average_loss:.3f}')
        
    # evaluation loop
    
# switch model to evaluation mode
model.eval()
    
correct = 0
total = 0
with torch.no_grad(): # turn off gradient tracking to speed up the process and reduce memory usage
    for inputs, labels in test_loader: # enumerate through the test_loder iterator
        inputs, labels = inputs.to(device), labels.to(device) # move them to the right spot
            
        outputs = model(inputs)
            
        # our predicted output is the output with the highest logic. This takes the max of all of our logits. dim=1 bc the shape is [batch_size], 1 dimensional
        _, predicted = torch.max(outputs, dim=1)
            
        total += labels.size(0)
        correct += (predicted == labels).sum().item() # predicted == labels is 1 if they are the same. sum() sums them all over a list, and .item()
    acc = correct * 100 / total
    print(f'Evaluated accuracy: {acc:.2f}%')
        