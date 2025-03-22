In the previous chapter, we've explored a simple MNIST digit recognition example with a training loop that looked like this:
```python
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
```
In this chapter, we'll rewrite it using [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html).
## Why PyTorch Lightning?
Implementing a training loop from scratch, like in the example above, was quite easy. However, there are some deep learning tricks that are missing, such as:
- Automatic mixed precision
- Gradient clamping and scaling
- Gradient accumulation  

Implementing these tricks in our own training loop from scratch is cumbersome and error-prone. In addition, their logic is pretty much fixed and will not change from one experiment to another. So the idea of PytorchLightning is to implement these once, and provide developers tools to inherit from some common class, only implementing the parts that are specific to the task at hand. As an extra bonus, lightning also provides some tools for speedup and debugging. [HuggingFace Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer) is another tool in similar category. In this blog series, we focus on lightning because it is used in [NeMo](https://github.com/NVIDIA/NeMo), industrial-grade repo with examples for DDP and model parallel training.   
## Using Lightning for our training
So, we've agreed that some tricks are needed, and their logic is fixed. What logic is not fixed?    

We most likely will need custom model and custom loss. Sometimes, we might also need custom optimizer setup - for example, when we need different learning rate for different parameter groups. These are exactly the things that we need to implement to get started with lightning.   

Let's assume we start with [simple MNIST example](../04_dist_training/train.py). This is how to implement a LightningModule:
```python
class MyLightningModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Net()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        imgs, labels = batch
        outputs = self.model(imgs)
        loss = F.nll_loss(outputs, labels)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.model.parameters(), lr=self.args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.args.gamma)
        return [optimizer], [scheduler]
```
We have `__init__`, `training_step`, and `configure_optimizers` methods.  

`training_step` receives a batch of data and a batch index, and returns the loss. `configure_optimizers` either creates a simple optimizer, or optimizer+scheduler, or something even more complicated. For example, you might override the frequency of lr steps. Refer to this [guide](https://lightning.ai/docs/pytorch/stable/common/optimization.html) for more details.  

To start the training:   
```python
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)

my_module = MyLightningModule(args)
trainer = L.Trainer(max_epochs=args.epochs)

trainer.fit(model=my_module, train_dataloaders=train_loader)
```
Refer to full [example](train.py) for more details.  

I can launch training with `python train.py`. If I do it on a machine with 8 gpus available, this is what I see in stdout:   
```
/traindata/maksim/miniconda3/envs/lightning/lib/python3.11/site-packages/lightning/fabric/plugins/environments/slurm.py:204: The `srun` command is available on your system but is not used. HINT: If your intention is to run Lightning on SLURM, prepend your python command with `srun` like so: srun python train.py ...
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
/traindata/maksim/miniconda3/envs/lightning/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/logger_connector/logger_connector.py:75: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `lightning.pytorch` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
Middle!
You are using a CUDA device ('NVIDIA H100 80GB HBM3') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/8
Initializing distributed: GLOBAL_RANK: 3, MEMBER: 4/8
Initializing distributed: GLOBAL_RANK: 4, MEMBER: 5/8
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/8
Initializing distributed: GLOBAL_RANK: 7, MEMBER: 8/8
Initializing distributed: GLOBAL_RANK: 5, MEMBER: 6/8
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/8
Initializing distributed: GLOBAL_RANK: 6, MEMBER: 7/8
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 8 processes

LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
LOCAL_RANK: 7 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
LOCAL_RANK: 3 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
LOCAL_RANK: 6 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
LOCAL_RANK: 2 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
LOCAL_RANK: 5 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]
LOCAL_RANK: 4 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7]

  | Name  | Type | Params | Mode
---------------------------------------
0 | model | Net  | 1.2 M  | train
---------------------------------------
1.2 M     Trainable params
0         Non-trainable params
1.2 M     Total params
4.800     Total estimated model params size (MB)
7         Modules in train mode
0         Modules in eval mode
/traindata/maksim/miniconda3/envs/lightning/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
Epoch 8: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [00:01<00:00, 99.42it/s, v_num=22]

Test set: Average loss: 0.0289, Accuracy: 9896/10000 (99%)
```
So, it automatically discovered that I have 8 gpus, started a multi-process group, and trained my model in DDP mode. Final accuracy is below what we had in previous chapter (`Accuracy: 9922/10000 (99%)`). It's also quite fast (1m vs 3m30s that we had in single-gpu mode before). WTF is happening? Let's unpack this step by step.   

First, we can add a print statement in the very beginning of our `train.py` script. If we do so, we'll see the multiprocessing logic of lighting: 
- We start the script as a normal python script
- When we reach the trainer.fit part, it forks 7 new processes. Our code gets executed 7 more times, not just the part after trainer.fit, but also before! Beware of this if your code has some side effects
- `torch.distributed.init_process_group` is called somewhere inside torch lightning. We didn't have to do it ourselves, nor did we need to place our models/data on gpu   

Finally, the reason accuracy is lower is because we need to adjust batch size. Our `batch_size` for dataloader is now effectively `micro_batch_size`, i.e., global batch size is now 8x larger. This is a rather underdocumented feature, I found some explanations around it in [this](https://www.restack.io/p/pytorch-lightning-answer-batch-size-multi-gpu-cat-ai) blog post.  

Here are some code modifications to take that into account:  
```python
parser.add_argument('--num-nodes', type=int, default=1, metavar='N',
                    help='number of nodes to train on (default: 2)')
parser.add_argument('--devices', type=int, default=8, metavar='N',
                    help='number of devices to train on (default: 8)')
world_size = args.num_nodes * args.devices
micro_batch_size = args.batch_size // world_size
train_kwargs = {'batch_size': micro_batch_size}
dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
# ...
trainer = L.Trainer(max_epochs=args.epochs, num_nodes=args.num_nodes, devices=args.devices)
```

Now restarting it with `python train.py --num-nodes 1 --devies 8` we get to a proper accuracy (`Accuracy: 9921/10000 (99%)`).   

To some up, what PyTorch Lightning does:
- It sets up process group automatically
- It discovers all available gpus and forks itself to spawn necessary processes
- It deals with distributed sampling and shuffling on its own. The only thing to take care of is to adjust batch size in the dataloader we provide
- It autodiscovers strategy to be DDP, and wraps the model accordingly  

Now we can fully benefit from all the functionality provided by lightning. For example, this is how we can switch to bf-16 mixed precision:  
```python
trainer = L.Trainer(max_epochs=args.epochs, strategy="ddp", num_nodes=args.num_nodes, devices=args.devices, precision="bf16-mixed")
```

## Multi-node training
Finally, as I've promised before, ddp should be easy to set up for multi-node, multi-gpu. Lightining makes it even easier by autodetecting if it has been launched in a context of a multi-node job. For example, with SLURM, we can write a `submission.sh` file like this:
```bash
#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=2           # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8   # This needs to match Trainer(devices=...)
#SBATCH --mem=0
#SBATCH --time=0-02:00:00

# activate conda env

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
srun python3 train.py --num-nodes 2 --devices 8
```
And run it with `sbatch submission.sh`. The only thing to take care of is to make sure that node/device counts in slurm request (`#SBATCH ...`) match that passed to the script. The rest will be done by lightning:  
- It will detect that its running in a context of a slurm multi-node job, disabling default forking mechanism
- It will get ranks/world size from slurm env variables
- It will set up dist process group, distributed sampling etc

## Up next
In the next few chapters, I will first set up a more real-world example of finetuning - finetuning retrieval embeddings for RAG application - and go over some of the training tricks that are available in lightning.

