## Intro
In this tutorial, we will learn about torch.dist module - a foundational library for cross-process communication, upon which most of modern distributed training frameworks are built. Tutorial roughly follows the [official tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) with some extra explanations from me. I do encourage ppl to read the original as well for some extra information. What we'll do today:  
- Run an example script with cross-process communication in torch.dist - single node, multi-process and multi-node.
- Talk a little bit about challenges of distributed communication and how torch.dist helps.

## Interprocess communication with torch.distributed.
One of the challenges of distributed training is to set up communication. In our previous chapter about torch.DataParallel, we only talked about single-process, multi-device parallelization, so it was a bit easier. What if we want multiple nodes? We'll have to set up communication somehow. And as soon as we do that, we'll need to care about so many things:  
- How do nodes connect to each other?
- How do we write actual network communication code, serialization, etc?
- How do we use our cluster topology optimally? For example, if we have several nodes with multiple gpus, how do we ensure that the communication layer knows that communication between two executors on the same node are faster than cross-node?
- If we need communication between two gpus on the same node, how do we avoid copying data on the host memory?
- If we use 3d party libraries for communication, how to we build and ship them along with our code? 

All of these are handled by torch.dist. Under the hood, it uses one of the three communication backends: `gloo`, `nccl` and `mpi` (though `mpi` is not shipped with pytorch and has to be installed separately). As a rule of thumb, `gloo` is preferred for cpu jobs while `nccl` is preferred for workloads relying heavily on gpu and CUDA tensors. 
### Starting the processes
First step in python code will always be `dist.init_process_group`. The purpose is two-fold: 
- Set up communication: set up master_addr, let the executors connect and become aware of each other.
- Let each executor know its rank and world size - to know which part of the job it is supposed to do.   

Here is a sample code to init the process group:  
```python
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist

def run(rank, size):
    """ Distributed function to be implemented later. """
    print("Running function. Current rank: ", rank)
    pass

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    if "MASTER_ADDR" not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    if "MASTER_PORT" not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    print("MASTER_ADDR: ", os.environ['MASTER_ADDR'])
    print("MASTER_PORT: ", os.environ['MASTER_PORT'])
    print("Starting init process group. Current rank: ", rank)
    dist.init_process_group(backend, rank=rank, world_size=size)
    print("Finished init process group. Current rank: ", rank)
    fn(rank, size)


if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    current_rank = int(os.environ["RANK"])
    init_process(current_rank, world_size, run)
```
We rely on some external env variables: WOLRD_SIZE and RANK for rank/world size, and MASTER_ADDR and MASTER_PORT to connect to the master. First, let's try to launch this locally:  
```bash
# launch in terminal
WORLD_SIZE=2 RANK=0 python main.py
```
This is what will be printed:
```
MASTER_ADDR:  127.0.0.1
MASTER_PORT:  29500
Starting init process group. Current rank:  0
```
We said that world_size is 2, so we expect two executors to connect. The first one launched will patiently wait for someone else to connect. Let's launch another one:
```bash
# launch in terminal
WORLD_SIZE=2 RANK=1 python main.py
```
From this script, we see:
```
MASTER_ADDR:  127.0.0.1
MASTER_PORT:  29500
Starting init process group. Current rank:  1
Finished init process group. Current rank:  1
Running function. Current rank:  1
```
And the first launched script also prints:
```
Finished init process group. Current rank:  0
Running function. Current rank:  0
```
So, after launching second script, they got connected and proceeded with their work.   

So far we launched them locally. But exactly the same code will work from different nodes - provided we can support them with connectivity through proper master addr/port. One way of doing this is with [slurm](https://slurm.schedmd.com/quickstart.html). Setting up slurm cluster is beyond the scope of this blog. Provided you already have a slurm cluster set up, launching distributed jobs is rather easy. For that, I use this script: 
```bash
# run.sh script
set -e # set fail on fail

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

python main.py
```
And I run it with:
```bash
srun -N 2 --time=1:00:00 bash run.sh
```
We'll see output like this:
```
MASTER_ADDR:  ip-172-19-245-211
MASTER_PORT:  12345
Starting process group. Current rank:  1
Finished process group. Current rank:  1
Running function. Current rank:  1
MASTER_ADDR:  ip-172-19-245-211
MASTER_PORT:  12345
Starting process group. Current rank:  0
Finished process group. Current rank:  0
Running function. Current rank:  0
```
Slurm's `srun` concatenates output from all the processes, but we can see that now MASTER_ADDR refers to a real ip address of some node, and the same communication layer allows executors to communicate across different machines. Despite the fact that in most of the tutorials we see `localhost` as a master address and single-node, multi-gpu setups, exactly the same code will work for multi-node. The only thing to be aware of in that case, as we'll see later, is to set up the input and the data properly so that all remote nodes can access it and each executor reads its own partition of data.   

Finally, another popular way of spawning several processes at once is `torchrun`. Without any modifications of our script:  
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 main.py
```
This will spawn two processes on the same node, and provide MASTER_ADDR, MASTER_PORT, WORLD_SIZE and RANK env variables automatically (though we did rely on these specific env variable names in our script!). `torchrun` is widely used to launch distributed training jobs, though most industrial-grade systems rely on frameworks like [pytorch lightning](https://lightning.ai/docs/pytorch/stable/index.html) or [huggingface trainer](https://huggingface.co/docs/transformers/main/en/trainer) to handle process management on a current node.   
### Point-to-point communication
Now let's demonstrate some actual communication examples. First, the basic primitive - point to point communication, aka send a message to a specific executor:  
```python
# replace run function above
def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])
```
Interface is quite neat. We decide what data to send (tensor, in our case), and to whom. We got our own rank from env variables (though it is always available via `dist.get_rank()`). Serialization and communication is entirely handled by `torch.dist`.    

One thing that might be quite mind-boggling at this point is which parts of the code get executed when and by whom. The answer is: there are 2 executors, executing exact same code, running in parallel. The only thing that distinguishes them is the rank. So we have to rely on this rank to actually differentiate which job is done by each executor.   

Point to point communication is rather niche and is not needed in most of distributed training scenarios, unless you are writing your own framework ([example](https://research.facebook.com/publications/accurate-large-minibatch-sgd-training-imagenet-in-1-hour/)) or some custom training algorithms. For everyday use, we'll usually rely on existing primitives like `all_reduce`.  

### Collective communication
```python
""" All-Reduce example."""
def run(rank, size):
    """ Simple collective communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])
```
`all_reduce` takes an operation (SUM, in our case), and applies it to all executors within a group. `group` is a way to subdivide executors into some sort of hierarchy, though usually it will just mean "all executors". The result of this operation will be:  
- All excecutors send their values for `tensor` somewhere
- All the values are summed up using some sort of algorithm and point-to-point communication. The actual algorithm (all-to-master, ring all-reduce, tree all-reduce, etc) depends on the backend as well as network topology, and is chosen automatically. For example, if we have several nodes with multiple gpus each, it makes sense to quickly sum up values within each node, and then let the nodes exchange their results.
- The result is sent back to each executor.
The call to `dist.all_reduce` is blocking, meaning that the code will wait for all the executors to finish their part of the operation. After this line, all executors will receive the final sum result in their `tensor` variable.   

Apart from all_reduce, there are [other](https://pytorch.org/tutorials/intermediate/dist_tuto.html#collective-communication) collective primitives, like max/mean/min and all_gather (get a list of all tensors from all executors).

### Debugging
It is often helpful to run a python script under debugger. Here is one way of how to do it:   
```python
def run(rank, size):
    tensor = torch.zeros(1)
    torch.distributed.breakpoint(0)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
        print("My rank: ", dist.get_rank())
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])
```
`torch.distributed.breakpoint(0)` works similar to `ipdb.set_trace()`, but with some extra additional features for distributed. It attaches debugger only for one rank (0 in our case, though you may choose to specify a different number), synchronizes all the executors with `torch.distributed.barrier()`, and sets a breakpoint. Running this script (either with `torchrun`, manually, or even with `srun`) will pause execution at a breakpoint and give you control, allowing to inspect variables, set further breakpoints etc. You will de-facto be debugging a single executor, but all the other executors will still be running as usual.   

It is a bit more tricky to set this up for multi-node jobs, like with slurm and `sbatch` which is typically used for launching distributed jobs.

## Outro
In this chapter, we've looked at basic primitives of torch.dist, how overall communication is being setup and even how to debug it. The things we talked about are rarely used in practice; most training use-cases rely on existing frameworks. But these foundations are helpful to understand how things work under the hood. 

In the next chapter, we'll look at how to actually implement distributed training with torch.dist (low-level) and torch.nn.parallel.DistributedDataParallel (high-level) on an example of MNIST digit recognition.