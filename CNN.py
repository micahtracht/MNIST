import torch.nn as nn
import torch.nn.functional as F
class SimpleCNN(nn.Module):
    def __init__(self):
        '''
        In this function, we're building the structure for the CNN.
        '''
        #This line lets us use the features from the parent nn.Module class to enable our class to do everything we want it to do.
        super(SimpleCNN, self).__init__() # could also use super().__init__(), but this is more explicit.
        
        # This line creates and will store the first layer of the CNN, the convolutional layer.
        # The input dimension is 1 (input depth = 1, grayscale). We use 10 filters, and they are all 5x5.
        # Output shape is (out_channels, out_size, out_size) where out_size = floor((input - kernel)/stride) + 1. With padding: out_size = floor((input + 2*padding - kernel)/stride) + 1
        # By including this as a part of the class, PyTorch will track its weights/biases, and include it in .to(), .eval(), .parameters(), and more.
        # num learnable params = out * in * kernel height * kernel width + out_channels (if bias is true)
        # This layer has 250 + 10 = 260 learnable parameters (250 weights, 10 biases). Dim in: [1, 28, 28]. Dim out: [10, 24, 24]
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = 5, bias = True) # bias defaults to True anyways
        
        # Similar to the line above, except now we're taking 10 inputs, one from each of the 10 outputs above.
        # Each filter looks at all 10 inputs simultaneously.
        # this layer has 5020 learnable parameters (5000 weights, 20 biases). Dim in: [10, 12, 12]. Dim out: [20, 8, 8]
        self.conv2 = nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size = 5, bias = True)
        
        # pooling layer. stride will default to kernel_sizes to ensure it is non-overlapping.
        # this layer helps with translational invariance by ensuring that small shifts are sort of aggregated together. It does this by taking the max value in each non-overlapping nxn (2x2 in this case) window.
        # also improves computational efficiency, since the size has been reduced by a factor of n, and helps with regularization.
        # Dim in: [10, 24, 24]. Dim out: [10, 12, 12]
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # same as above
        # Dim in: [20, 8, 8]. Dim out: [20, 4, 4]
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # fc stands for fully connected
        # in_features = 320 because dim in = [20, 4, 4], which has 20x4x4 = 320 total features. out_features is picked to be in a healthy middle ground.
        # Too small, it'll underfit. Too large, it'll overfit. 50 is a good balance for a small dataset like MNIST.
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        
        # 50 in_features from the 50 outputted from fc1. 10 output features since we're guessing 10 things: the digits 0-9.
        self.fc2 = nn.Linear(in_features=50, out_features=10)
    
    # x has shape [batch_size, 1, 28, 28], since it is a grayscale (depth 1), 28x28 pixel image, and we are using batches. 4D tensor.
    def forward(self, x):
        x = self.conv1(x) # simply calls conv1, as defined above. x now has dimensions [batch_size, 10, 24, 24], the same as the output dimensions of conv1.
        x = F.relu(x) # applies the ReLU: f(x) = max(0, x). This allows for our process to learn nonlinear relationships. Otherwise, it'd always be linear, and it'd be a glorified least squares.
        # x still has dimensions [batch_size, 10, 24, 24], as ReLU does not change the dimensions.
        x = self.pool1(x) # x now has dimensions [batch_size, 10, 12, 12] after being pooled.
        
        x = self.conv2(x) # x now has dimensions [batch_size, 20, 8, 8]
        x = F.relu(x) # second ReLU
        x = self.pool2(x) # second pooling. x now has dimensions [batch_size, 20, 4, 4]
        
        # 320 = 20 * 4 * 4 = number of elements in x. x.view() reshapes how x is stored in memory so it goes from being a 3D tensor to 1d. 
        # the -1 tells it to infer the size of the batches, and the 320 tells it there should be 320 elements in x.
        x = x.view(-1, 320) # x now has dimensions [batch_size, 320]. This is the flattening step.
        
        x = self.fc1(x) # x now has dimensions [batch_size, 50]
        x = F.relu(x)
        x = self.fc2(x) # x now has dimensions [batch_size, 10]
        
        return x # x shape: [batch_size, 10]