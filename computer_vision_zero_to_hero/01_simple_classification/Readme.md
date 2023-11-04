## Intro
Today, we are going to train a simple neural network with just some dense layers to classify images in the MNIST dataset. The dataset contains low-res (28*28) images of digits.  
The purpose of this post is to showcase that even with the simple network based just on dense layers, without convolutions/transformers and other such stuff, it is possible to build quite an efficient classifier.  
We pretty much follow [this](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) guide with some modifications and explanations.  
We are going to write the model in PyTorch, one of the two major neural network frameworks (other one being Tensorflow). Cool thing about PyTorch is that installation is rather easy. I've just spent a few days trying to perform installation steps for Tensorflow. It wouldn't work on Mac M1 at all, and making it work on Ubuntu with amd chip took a few days of reading forums, trying different solutions from the internet, and debugging. PyTorch worked out of the box on Mac M1, and can even exploit Apple's "Metal" framework to train models on gpu.
## Setup and getting data
I am running my code in jupyter notebook. 
If you haven't already, set up these dependencies:
```
%pip install torch
%pip install torchvision
```
Do necessary imports:
```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```
And load the data:
```python
# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```
Here, we just load it from the Internet using some magic function from PyTorch. Popular datasets are usually available like this in PyTorch, Tensorflow, and other frameworks, saving us the hassle of having to deal with data parsing and conversion.
It's quite hard to find the actual origins of the dataset now, but here is another link to it on [hugginface](https://huggingface.co/datasets/mnist) - it lets you explore it a bit more, looking at the actual images and labels.  
Now, we convert these "datasets" into `DataLoader`s:
```python
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```
```
Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64
```
This of a `DataLoader` as a common API that allows you to iterate over features and labels from the dataset. Our current case is trivial - all our data is loaded into memory, it fits on one machine, and training is done on the same machine. DataLoader api can be used for more complex cases - data not fitting into memory entirely and distributed accross files instead, or multiple training nodes reading data from the cloud in parallel.  
Finally, we can simply iterate over the `DataLoader` to get batched examples: features and labels.  
Features have dimensions of `batch_size` * `colour` * `height` * `width`. Batch size is traditionally the first dimension; colour in our example has a size of 1, i.e. there is no actual colour, just shades of grey. We still have it as an extra dimension for compatibility with coloured images - they will have size of 3, for RGB encoded images.  

Our current example is rather straightforward, but I find it useful to always make an extra step and take a look at the data and all intermediate modeling steps as well.  
Here is how we can print the features (i.e., the actual image) for one example:
```python
X[0]
```
```
tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000],
          ...
          ]])
```
It's quite hard to understand what is going on here because of the formatting, so let's try to fix that:
```python
x = X[0].squeeze()
for i in range(len(x)):
    for j in range(len(x[0])):
        q = x[i][j].item()
        w = 0
        if q > 0:
            w = 1
        print("{}".format(w), end="")
    print()
```
```
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000001111110000000000000000
0000001111111111111111000000
0000001111111111111111000000
0000000000011111111111000000
0000000000000000001111000000
0000000000000000011110000000
0000000000000000011110000000
0000000000000000111100000000
0000000000000000111100000000
0000000000000001111000000000
0000000000000001110000000000
0000000000000011110000000000
0000000000000111100000000000
0000000000001111100000000000
0000000000001111000000000000
0000000000011111000000000000
0000000000011110000000000000
0000000000111110000000000000
0000000000111110000000000000
0000000000111100000000000000
0000000000000000000000000000
```
We need `tensor.squeeze()` to get rid of trivial dimensions, i.e. turn a tensor with shape (1, 28, 28) into (28, 28).  
We replaced everything "slightly gray" with 1, and we can already sort of see what this was supposed to represent - a digit "7".  
And here is how we can actually plot it:
```python
import matplotlib.pyplot as plt
def show_image(image, label):
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Label: {label}')
    plt.show()
show_image(X[0], y[0])
```
![image of number 7](image1.png "Title")  
Now let's deal with the devices:
```python
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
```
Having a gpu might accelerate training by 100x, but it can be quite painful to install all the necessary drivers. *CUDA* is a gpu processing platform by NVidia, and it is typically used on all intel/amd based computers. "mps" stands for "metal performance shaders" - this is Apple's analogue to CUDA. In my case, I am running this on M1 mac, and "mps" worked out of the box.  
Now, we define the model:
```python
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```
We define the model in 2 steps: `init` to set up the layer configuration, and `forward` to call those layers, thus initializing the entire computation graph, from inputs to outputs.  
We call `something.to(device)` to place it on the specified device. This is needed for gpu/mps acceleration, but be careful - in order for the code to work, everything used in a certain command needs to be placed on the same device.  
Once again, we can print all the intermediate results:  
```python
print(X.shape)
f = nn.Flatten()(X)
f.shape
```
```
torch.Size([64, 1, 28, 28])
torch.Size([64, 784])
```
`nn.Flatten()` is used to flatten all the dimensions except the batch dimension, turning a "square" of 1 * 28 * 28 into a single array with length 784.
We can do the same with lineaer layer:
```python
p = nn.Linear(28*28, 512)(f)
p.shape
```
```
torch.Size([64, 512])
```
It's easy to think about dense (linear) layers as *projectors* from size A to size B, leaving batch dimension unchanged.  
Finally, let's check that the actual model also works:
```python
preds = model(X[0].to(device))
preds
```
```
tensor([[ 0.0327, -0.0553, -0.0298, -0.0272,  0.0073,  0.0156,  0.0305, -0.0962,
         -0.0051, -0.0906]], device='mps:0', grad_fn=<LinearBackward0>)
```
Since we've already placed our model on the "mps" device, we have to do the same to the features vector *X[0]*. After several dense layers, we project initial 28*28 dimension into 512, then into 512 again, then into 10. These final 10 outputs will be used to get the actual probabilities of the classes. In MNIST digits dataset, class indices 0-9 actually somewhat correspont to class labels: "0", "1", "2", ... This is just a happy coincidence; in FashionMNIST, for example, class indices will still be numbers 0-9, but 0 will mean "T-shirt", 1 - "trouser" and so on.  

Another important thing - parameter initialisation:
```python
list(model.linear_relu_stack[0].parameters())
```
```
[Parameter containing:
 tensor([[-0.0014,  0.0336,  0.0182,  ..., -0.0318, -0.0343, -0.0282],
         [-0.0036, -0.0323, -0.0093,  ..., -0.0015, -0.0115, -0.0050],
         [ 0.0131,  0.0263, -0.0080,  ...,  0.0056, -0.0090,  0.0225],
         ...,
         [ 0.0236,  0.0344,  0.0041,  ...,  0.0132, -0.0329, -0.0338],
         [ 0.0320,  0.0177, -0.0015,  ..., -0.0104, -0.0307, -0.0196],
         [-0.0234, -0.0330, -0.0115,  ...,  0.0228, -0.0003, -0.0114]],
        requires_grad=True),
        ...
]
```
Note that initial values for parameters are not 0, and not random in [0, 1). Zero parameters will make all predictions trivial, and the model will not be able to properly train from that point - in effect, getting "stuck" in the local maximum of loss function. Initializing parameters in [0, 1) will cause the prediction of the model to "explode" to a really large number, because input is multiplied by the matrix of parameters. The bigger the matrix, the bigger the result would be.  
Instead, PyTorch default is random in (-sqrt(N), +sqrt(N)) where N is the number of input features - this makes sure that if this layer receives input in [0, 1), the output will be in roughly the same interval, thus allowing us to stack many such layers on top of each other without fear of final output disproportionally exploding or shrinking.  

Now we define a loss and optimiser:
```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn(preds[0].to(device), y[0].to(device))
```
```
tensor(2.3780, device='mps:0', grad_fn=<NllLossBackward0>)
```
Since we want to solve classification problem, we use CrossEntropy for loss. SGD stands for "stochastic gradient descent".  
I'm also printing out actual loss function value on one example. This is yet another check - making sure that we are doing everything correctly, like model output format corresponding to requirements of loss function input.  

Next, we define our training function:
```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```
`model.train()` simply sets the model in the *training mode*. On top of getting prediction out of the model, calling the model now also triggers the autograd machinery that will let us calculate gradients, but works a bit slower.
We then iterate over batches in the dataset; calculate predictions for the entire batch, and the loss function.  
`loss.backward()` computes the gradients. They were already "*connected*" to the optimizer, so calling `optimizer.step()` performs one step of optimisation.  
We also print out train loss every once in a while.  
Apart from train loss, lets compute test loss:
```python
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```
And the final train loop:
```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```
In each epoch, we iterate over entire train dataset once, and then compute test loss and accuracy. 
```
Test Error: 
 Accuracy: 11.9%, Avg loss: 2.315746 

Epoch 1
-------------------------------
loss: 2.320072  [   64/60000]
loss: 2.281309  [ 6464/60000]
loss: 2.270785  [12864/60000]
loss: 2.220879  [19264/60000]
loss: 2.213635  [25664/60000]
loss: 2.202037  [32064/60000]
loss: 2.129568  [38464/60000]
loss: 2.146335  [44864/60000]
loss: 2.097662  [51264/60000]
loss: 2.075108  [57664/60000]
Test Error: 
 Accuracy: 64.7%, Avg loss: 2.051794 

Epoch 2
-------------------------------
loss: 2.059916  [   64/60000]
loss: 2.008694  [ 6464/60000]
loss: 2.020241  [12864/60000]
loss: 1.905036  [19264/60000]
loss: 1.925415  [25664/60000]
loss: 1.910556  [32064/60000]
loss: 1.800620  [38464/60000]
loss: 1.871886  [44864/60000]
loss: 1.773053  [51264/60000]
loss: 1.727718  [57664/60000]
Test Error: 
 Accuracy: 72.1%, Avg loss: 1.698042 

Epoch 3
-------------------------------
loss: 1.718040  [   64/60000]
loss: 1.630872  [ 6464/60000]
loss: 1.664411  [12864/60000]
loss: 1.499201  [19264/60000]
loss: 1.524873  [25664/60000]
loss: 1.503511  [32064/60000]
loss: 1.374756  [38464/60000]
loss: 1.506106  [44864/60000]
loss: 1.383151  [51264/60000]
loss: 1.321513  [57664/60000]
Test Error: 
 Accuracy: 75.9%, Avg loss: 1.283851 

Epoch 4
-------------------------------
loss: 1.324238  [   64/60000]
loss: 1.211452  [ 6464/60000]
loss: 1.271265  [12864/60000]
loss: 1.116692  [19264/60000]
loss: 1.126974  [25664/60000]
loss: 1.117098  [32064/60000]
loss: 1.015542  [38464/60000]
loss: 1.180133  [44864/60000]
loss: 1.078986  [51264/60000]
loss: 1.023408  [57664/60000]
Test Error: 
 Accuracy: 80.1%, Avg loss: 0.976671 

Epoch 5
-------------------------------
loss: 1.038096  [   64/60000]
loss: 0.917698  [ 6464/60000]
loss: 0.981325  [12864/60000]
loss: 0.875240  [19264/60000]
loss: 0.867240  [25664/60000]
loss: 0.860575  [32064/60000]
loss: 0.789301  [38464/60000]
loss: 0.954533  [44864/60000]
loss: 0.889059  [51264/60000]
loss: 0.847700  [57664/60000]
Test Error: 
 Accuracy: 82.5%, Avg loss: 0.790373 
```
First thing we get from this - train and test loss history. This is a rather crude way to do it, and later we'll look at more convenient ways, like Tensorboard. But for now this will do. Train loss will always go down with each new epoch. Test loss might start to go up at some point, indicating overfitting. For now it seems more like underfitting.  

We get accuracy of 82.5%. Of course, that is certainly not ideal result, but it is fairly better than random prediction (we'd expect it to have roughly 10% accuracy). It's also interesting to get such a good result with such a simple model. Certainly makes for a good baseline.  

Let's also check prediction on the actual number "seven" we've seen before:
```python
preds = model(X[0].to(device))
preds
```
```
tensor([[-0.4013, -2.2614, -1.6052, -0.1392,  0.8867, -0.1384, -2.3127,  4.3744,
         -0.4752,  2.1780]], grad_fn=<AddmmBackward0>)
```
Logit for index 7 is indeed the biggest, so the model can correctly classify our example!