In this chapter, we will first train a simple digit classification model (MNIST) following [example](https://github.com/pytorch/examples/blob/main/mnist/main.py). Then, we'll write our own Data Parallel code to train the same model in distributed setup; finally, we'll show how same functionality can be achieved using PyTorch Distributed Data Parallel (DDP).
## Simple training
For full working example, refer to [train.py](train.py).  
First, let's take a look at the data:   
```python
from torchvision import datasets, transforms
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
example = dataset1[0]
img, label = example
img.shape, label
```
```
(torch.Size([1, 28, 28]), 5)
```
We use `torchvision.datasets` module to get the data. It downloads the data automatically, and returns it in `torch.Datasets` format - something you can index into using square brackets. By default, it will return either a PIL image or a numpy array. We provide extra "transforms" argument to convert it to a tensor and normalize it. Each training example contains an image (28x28, all images are equal size) and a label - the actual digit depicted on the image, 0-9.   

Next, we define a data loader:
```python
train_kwargs = {'batch_size': args.batch_size}
cuda_kwargs = {'num_workers': 1,
    'pin_memory': True, # prepare data to be moved to gpu memory
    'shuffle': True}
train_kwargs.update(cuda_kwargs)

train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
for batch in train_loader:
    break
imgs, preds = batch
imgs.shape, preds.shape
```
```
(torch.Size([64, 1, 28, 28]), torch.Size([64]))
```
Data loader iterates over the dataset by generating specific indices which need to be loaded. With `shuffle=True`, indices will be shuffled. Dataloader also unites examples into batches. In our case, all images are of the same shape, so no custom `collate_fn` is needed. As a result, we get a batch of images and labels.   

Next, our super SOTA image classification model:
```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```
Next, we define a model, put it on GPU, and train it:
```python
model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    scheduler.step()
```
Here is how train function looks like:
```python
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
```
`train` function trains for one epoch, i.e., iterates over entire dataset once. It applies the model, computes loss, computes gradients with `loss.backwward()`, and updates model parameters with `optimizer.step()`.  
We can run the training with `python train.py`, and measure training time with `time python train.py`. The script will also print test metrics. Here is what I got with default options:  
```
Test set: Average loss: 0.0256, Accuracy: 9922/10000 (99%)


real    2m23.499s
user    3m16.891s
sys     0m13.376s
```
We can also check that only one gpu is used with `nvidia-smi` during training. In my case, on H100, the actual gpu usage is around 5-7%, because the model is extremely small.
## Writing our own Distributed Data Parallel
First, let's add our init process code at the very beginning of the script:
```python
def init_process(backend='nccl'):
    """ Initialize the distributed environment. """
    world_size = int(os.environ["WORLD_SIZE"])
    current_rank = int(os.environ["RANK"])
    if "MASTER_ADDR" not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    if "MASTER_PORT" not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    print("MASTER_ADDR: ", os.environ['MASTER_ADDR'])
    print("MASTER_PORT: ", os.environ['MASTER_PORT'])
    print("Starting init process group. Current rank: ", current_rank)
    dist.init_process_group(backend, rank=current_rank, world_size=world_size)
    print("Finished init process group. Current rank: ", current_rank)

init_process()
```
We fetch all required info from env variables.  

Recall that when using `torch.distributed` or any of the libraries that rely on it, we are supposed to launch the same script many times? This means that we have to make sure that script code is reading only its own portion of data. This is rather trivial to write, but might be tricky to debug and spot problems. Worst-case scenario - some data is omitted or all executors train on the same data - will not be obvious to spot. There will be no errors reported, but the model will not train correctly. That's why it is useful to monitor resulting metrics when writing code like this.   

We will rely on `torch.utils.data.distributed.DistributedSampler`. It basically does all the job for us:   
```python
train_sampler = torch.utils.data.distributed.DistributedSampler(dataset1)
train_loader = torch.utils.data.DataLoader(dataset1, sampler=train_sampler, **train_kwargs)
```
Under the hood, it does roughly the following:   
- Reads current rank and world size. 
- Splits data into `world_size` non-intersecting chunks
- When iterating over the dataloader, for current executor, it will provide only its own portion of data.   

It can also do shuffling. It is rather non-trivial to do stable shuffling in distributed context, so to make it work, we need to manually let the sampler know when new epoch has started. Here is our modified epoch iteration loop:
```python
for epoch in range(1, args.epochs + 1):
    train_sampler.set_epoch(epoch)
    train(args, model, device, train_loader, optimizer, epoch)
    scheduler.step()
```
It works roughly like this:
- At the beggining of each epoch, sampler sets a fixed, predetermined seed for pseudo-random shuffling RNG
- Every epoch, the seed is different, so shuffling will be different epoch-to-epoch
- Each executor has its own sampler, but thanks to fixed seed and pseudo-randomness, all executors are deterministic. This ensures that data partitions stay non-intersecting

Next, we need to write actual gradient exchange code. First, where can we get the actual gradients? In our training code, we compute them with `loss.backward()` and apply with `optimizer.step()`, but there are no explicit gradients being passed around. This is how we can take a look at actual gradients:
```python
for i, param in enumerate(model.parameters()): # fetch params of the first layer
    break
print(param.shape)
param.grad is None
```
```
torch.Size([32, 1, 3, 3])
True
```
We can get all model parameters with `model.parameters()`. Each parameter has a `grad` field, which is `None` by default, either before the gradients were computed, or after `optimizer.zero_grad()` was called.   
After we compute the loss and the gradients, this field is no longer `None`:
```python
output = model(data)
loss = F.nll_loss(output, target)
loss.backward()
print(param.grad is None)
param.grad.shape
```
```
False
torch.Size([32, 1, 3, 3])
```
As expected, gradients now exist, and their shape is exactly the same as the parameter shape. Here is how we can exchange gradients between executors:  
```python
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
```
We iterate over all parameters of the model. `dist.all_reduce` call computes sum of the gradients for current tensor across all executors, and stores the result in `param.grad.data`. Finally, we divide the result by the world size to get the average gradient. This is also what happened in the original, single-executor code. If we look at the doc for [nll_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html), loss values per every element in the batch are also being averaged out. This is the default behaviour for DDP, but if your code relies on sum of losses instead of average, this code will not work correctly.  

Here is the modified training loop:
```python
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        average_gradients(model)
        optimizer.step()
```
We insert `average_gradients` call right after `loss.backward()`, and before `optimizer.step()`. This ensures that gradients are averaged out before they are used to update model parameters. This call is also the *synchronization point* for all executors. If current executor is done with gradient computation on its portion of data, it will wait for all other executors to complete before proceding with `optimizer.step`.   

There are a couple of subtle details in this code to discuss.   
First, we first wait for all executors to compute all gradients, and then we iterate on the model layer-by-layer, starting from the first layer of the model, and exchange gradients before moving to the next layer. One might prefer instead to   
- Do exchange steps for all gradients at once to simplify communication OR
- Start exchanging gradients for the last layers of the model once they got computed, and do that in parallel to computing gradients for earliest layers.   

Refer to this [post](https://sebarnold.net/dist_blog/) to find out more about tricks like this. `torch.nn.parallel.DistributedDataParallel` is of course a better way of duing data parallel training that takes care of all these details automatically.

Another subtle thing is the state of the optimizer. The exchange happens before `optimizer.step()`. That means that all executors will have exactly equal gradients before applying them, but will compute updates, L1/L2 momentum, etc, each on its own. This is the same as we had for `DistributedSampler` - every executor executes the same code on its own and we hope to arrive at the same result, except that now we are dealing with floating point numbers, so rounding errors might cause different executors to converge to different values eventually.  

Finally, apart from model gradients, we might want to exchange states for other things like `BatchNorm` buffer. This is not done in our toy code, but it is handled in`torch.nn.parallel.DistributedDataParallel`.   

As a final touch, we also want to make sure that every executor works on its own gpu:
```python
device = torch.device(f"cuda:{dist.get_rank()}")
```
This simple code will work for single-node, multi-gpu setup. For multi-node we'll have to come up with something different, like 
```python
device = torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")
```

One of the ways to launch training in distributed setup:  
```bash
time torchrun --standalone --nnodes=1 --nproc-per-node=2 train_dist.py
```
```
Test set: Average loss: 0.0262, Accuracy: 9924/10000 (99%)

real    1m31.293s
user    3m46.595s
sys     0m20.381s
```
We can see that accuracy is still the same, but training is faster. We can also observe that 2 gpus are being used with `nvidia-smi`. 
## torch.nn.parallel.DistributedDataParallel
The above code was mainly for demonstration purposes. In practice, using `torch.nn.parallel.DistributedDataParallel` is both easier and better. It will require just a few modifications to the original, single-process code.   

```python
# Make sure that every process uses its own gpu
device = torch.device(f"cuda:{dist.get_rank()}")
# ...
# wrap the model in DDP
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])
# ...
# Make sure that each executor reads its own portion of data
train_sampler = torch.utils.data.distributed.DistributedSampler(dataset1)
train_loader = torch.utils.data.DataLoader(dataset1, sampler=train_sampler, **train_kwargs)
```
We'll also need to launch multiple processes, with `torchrun` or otherwise.   

To sum op the differences between `DataParallel` and `DistributedDataParallel`:   
- DataParallel only needs one process to be started. Splitting data across gpus, and synchronization of gradients is handled automatically, within model forward pass of the wrapped model.
- DistributedDataParallel requires multiple processes to be started.
- Code for data loading will have to provide executor with its own portion of data. This is easy with `DistributedSampler`.
- With DDP, multi-node, multi-gpu training will also work, though might require slight modifications to device handling and data loading.   

In the next chapter, we'll take a look at Torch Lightning - one of the high-end libraries for distributed training. Apart from distribution, it also takes care of a lot of things regarding training loop.

