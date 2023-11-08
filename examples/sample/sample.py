# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import gpu_affinity
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch GPU affinity sample',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--local_rank',
        type=int,
        default=os.getenv('LOCAL_RANK', 0),
        help='Used for multi-process training.',
    )
    parser.add_argument(
        '--mode',
        type=gpu_affinity.Mode,
        default=gpu_affinity.Mode.UNIQUE_CONTIGUOUS,
        choices=list(gpu_affinity.Mode),
        help='affinity mode',
    )
    parser.add_argument(
        '--multithreading',
        type=gpu_affinity.Multithreading,
        default=gpu_affinity.Multithreading.ALL_LOGICAL,
        choices=list(gpu_affinity.Multithreading),
        help='multithreading mode',
    )
    parser.add_argument(
        '--scope',
        type=gpu_affinity.Scope,
        default=gpu_affinity.Scope.NODE,
        choices=list(gpu_affinity.Scope),
        help='affinity scope',
    )
    parser.add_argument(
        '--min_physical_cores',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--max_physical_cores',
        type=int,
        default=None,
    )
    parser.add_argument(
        "--balanced",
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    nproc_per_node = torch.cuda.device_count()
    affinity = gpu_affinity.set_affinity(
        args.local_rank,
        nproc_per_node,
        mode=args.mode,
        scope=args.scope,
        multithreading=args.multithreading,
        balanced=args.balanced,
        min_physical_cores=args.min_physical_cores,
        max_physical_cores=args.max_physical_cores,
    )
    print(
        f'rank {args.local_rank}: '
        f'core affinity: {sorted(affinity)}, '
        f'num logical cores: {len(affinity)}'
    )


if __name__ == "__main__":
    main()
