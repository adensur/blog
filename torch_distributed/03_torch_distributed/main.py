#!/usr/bin/env python
import os
import torch
import torch.distributed as dist

def run(rank, size):
    """ Simple collective communication. """
    tensor = torch.ones(1)
    torch.distributed.breakpoint(0)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print('Rank ', rank, ' has data ', tensor[0])

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
    print("process start")
    world_size = int(os.environ["WORLD_SIZE"])
    current_rank = int(os.environ["RANK"])
    init_process(current_rank, world_size, run)
    print("Process exit")