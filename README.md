# GPU Affinity

GPU Affinity is a package to automatically set the CPU process affinity to
match the hardware architecture on a given platform. Setting the proper
affinity usually improves and stabilizes the performance of deep learning
workloads.

This package is meant to be used for multi-process single-device workloads
(there are multiple training processes, and each process is running on a single
GPU), which is typical for multi-GPU training workloads using
`torch.nn.parallel.DistributedDataParallel`.

## Features
* respects restrictions set by external environment (e.g. from `taskset` or
  from `docker run --cpuset-cpus`)
* correctly handles hyperthreading siblings
* supports `nvml` scope modes (`NUMA` and `SOCKET`)
* multiple affinity mapping modes, default arguments tuned for training
  workloads on DGX machines
* automatically sets "balanced" affinity (an equal number of physical CPU cores
  is assigned to each process)
* supports device reordering with `CUDA_VISIBLE_DEVICES` environment variable

## Installation

### Using pip
Install the package with `pip` directly from `github`.

```
pip install git+https://github.com/NVIDIA/gpu_affinity
```

### Manual installation

```
git clone https://github.com/NVIDIA/gpu_affinity
cd gpu_affinity
pip install .
```

## Usage

Install the package and call `gpu_affinity.set_affinity(gpu_id,
nproc_per_node)` function at the beginning of the `main()` function, right
after command-line arguments were parsed, and before any function which performs
significant compute or creates a CUDA context.

Warning: `gpu_affinity.set_affinity()` should be called once in every process
using GPUs. Calling `gpu_affinity.set_affinity()` more than once per process
may result in errors or a suboptimal affinity setting.

## Documentation

```
def set_affinity(
    gpu_id: int,
    nproc_per_node: int,
    *,
    mode: Mode = Mode.UNIQUE_CONTIGUOUS,
    scope: Scope = Scope.NODE,
    multithreading: Multithreading = Multithreading.ALL_LOGICAL,
    balanced: bool = True,
    min_physical_cores: int = 1,
    max_physical_cores: Optional[int] = None,
):
    r'''
    The process is assigned with a proper CPU affinity that matches CPU-GPU
    hardware architecture on a given platform. Usually, setting proper affinity
    improves and stabilizes the performance of deep learning training
    workloads.

    This function assumes that the workload runs in multi-process single-device
    mode (there are multiple training processes, and each process is running on
    a single GPU). This is typical for multi-GPU data-parallel training
    workloads (e.g., using `torch.nn.parallel.DistributedDataParallel`).

    Available affinity modes:
    * `Mode.ALL` - the process is assigned with all available physical CPU
        cores recommended by pynvml for the GPU with a given id.
    * `Mode.SINGLE` - the process is assigned with the first available physical
        CPU core from the list of all physical CPU cores recommended by pynvml
        for the GPU with a given id (multiple GPUs could be assigned with the
        same CPU core).
    * `Mode.SINGLE_UNIQUE` - the process is assigned with a single unique
        available physical CPU core from the list of all CPU cores recommended
        by pynvml for the GPU with a given id.
    * `Mode.UNIQUE_INTERLEAVED` - the process is assigned with a unique subset
        of available physical CPU cores from the list of all physical CPU cores
        recommended by pynvml for the GPU with a given id, cores are assigned
        with interleaved indexing pattern
    * `Mode.UNIQUE_CONTIGUOUS` - (the default mode) the process is assigned
        with a unique subset of available physical CPU cores from the list of
        all physical CPU cores recommended by pynvml for the GPU with a given
        id, cores are assigned with contiguous indexing pattern

    `Mode.UNIQUE_CONTIGUOUS` is the recommended affinity mode for deep learning
    training workloads on NVIDIA DGX servers.

    Available affinity scope modes:
    * `Scope.NODE` - sets the scope for pynvml affinity queries to NUMA node
    * `Scope.SOCKET` - sets the scope for pynvml affinity queries to processor
        socket

    Available multithreading modes:
    * `Multithreading.ALL_LOGICAL` - assigns the process with all logical cores
        associated with a given corresponding physical core (i.e., it
        automatically includes all available hyperthreading siblings)
    * `Multithreading.SINGLE_LOGICAL` - assigns the process with only one
        logical core associated with a given corresponding physical core (i.e.,
        it excludes hyperthreading siblings)

    Args:
        gpu_id (int): index of a GPU, value from 0 to `nproc_per_node` - 1
        nproc_per_node (int): number of processes per node
        mode (gpu_affinity.Mode): affinity mode (default:
            `gpu_affinity.Mode.UNIQUE_CONTIGUOUS`)
        scope (gpu_affinity.Scope): scope for retrieving affinity from pynvml
            (default: `gpu_affinity.Scope.NODE`)
        multithreading (gpu_affinity.Multithreading): multithreading mode
            (default: `gpu_affinity.Multithreading.ALL_LOGICAL`)
        balanced (bool): assigns equal number of physical cores to each
            process, it affects only `gpu_affinity.Mode.UNIQUE_INTERLEAVED` and
            `gpu_affinity.Mode.UNIQUE_CONTIGUOUS` affinity modes (default:
            `True`)
        min_physical_cores (int): the intended minimum number of physical cores
            per process, code raises GPUAffinityError if the number of
            available cores is less than `min_physical_cores` (default: 1)
        max_physical_cores: the intended maxmimum number of physical cores per
            process, the list of assigned cores is trimmed to the first
            `max_physical_cores` cores if `max_physical_cores` is not None
            (default: `None`)

    Returns a set of logical CPU cores on which the process is eligible to run.

    WARNING: On NVIDIA DGX A100, only half of the CPU cores have direct access
    to GPUs. set_affinity with `scope=Scope.NODE` restricts execution only to
    the CPU cores directly connected to GPUs. On DGX A100, it will limit the
    execution to half of the CPU cores and half of CPU memory bandwidth (which
    may be fine for many DL models). Use `scope=Scope.SOCKET` to use all
    available DGX A100 CPU cores.

    WARNING: Intel's OpenMP implementation resets affinity on the first call to
    an OpenMP function after a fork. It's recommended to run with the following
    env variable: `KMP_AFFINITY=disabled` if the affinity set by
    `gpu_affinity.set_affinity` should be preserved after a fork (e.g. in
    PyTorch DataLoader workers).

    Example:

    import argparse
    import os

    import gpu_affinity
    import torch


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--local_rank',
            type=int,
            default=os.getenv('LOCAL_RANK', 0),
        )
        args = parser.parse_args()

        nproc_per_node = torch.cuda.device_count()

        affinity = gpu_affinity.set_affinity(args.local_rank, nproc_per_node)
        print(f'{args.local_rank}: core affinity: {affinity}')


    if __name__ == '__main__':
        main()

    Launch the example with:
    python -m torch.distributed.run --nproc_per_node <#GPUs> example.py
    '''
```
