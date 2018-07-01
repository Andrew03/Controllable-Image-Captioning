import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run(rank, size, args):
	""" Distributed function to be implemented later. """
	print("{}, {}".format(rank, size))

def average_gradients(decoder):
    size = float(dist.get_world_size())
    for param in decoder.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def init_processes(rank, size, args, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, args)


if __name__ == "__main__":
    size = 3
    processes = []
    args = None
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, args, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
