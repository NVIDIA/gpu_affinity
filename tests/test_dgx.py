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

import itertools
import unittest
import unittest.mock

import gpu_affinity


class MockAffinity:
    @classmethod
    def reset(cls, num_cores):
        cls._affinity = set(range(num_cores))

    @classmethod
    def set(cls, pid, mask):
        assert pid == 0
        cls._affinity = set(mask)

    @classmethod
    def get(cls, pid):
        assert pid == 0
        return cls._affinity


def cores_set(start, stop, step=1, offset=None):
    the_list = list(range(start, stop, step))
    if offset is not None:
        the_list += list(range(start + offset, stop + offset, step))
    return set(the_list)


class DGX1VDevice:
    def __init__(self, device_idx):
        super().__init__()
        self.device_idx = device_idx

    def get_cpu_affinity(self, scope):
        if scope in {gpu_affinity.Scope.SOCKET, gpu_affinity.Scope.NODE}:
            if self.device_idx in {0, 1, 2, 3}:
                return list(range(0, 20)) + list(range(40, 60))
            elif self.device_idx in {4, 5, 6, 7}:
                return list(range(20, 40)) + list(range(60, 80))
            else:
                raise RuntimeError('Unknown device_idx')
        else:
            raise RuntimeError('Unknown scope')


class DGX2HDevice:
    def __init__(self, device_idx):
        super().__init__()
        self.device_idx = device_idx

    def get_cpu_affinity(self, scope):
        if scope in {gpu_affinity.Scope.SOCKET, gpu_affinity.Scope.NODE}:
            if self.device_idx in set(range(0, 8)):
                return list(range(0, 24)) + list(range(48, 72))
            elif self.device_idx in set(range(8, 16)):
                return list(range(24, 48)) + list(range(72, 96))
            else:
                raise RuntimeError('Unknown device_idx')
        else:
            raise RuntimeError('Unknown scope')


class DGXA100Device:
    def __init__(self, device_idx):
        super().__init__()
        self.device_idx = device_idx

    def get_cpu_affinity(self, scope):
        if scope == gpu_affinity.Scope.SOCKET:
            if self.device_idx in {0, 1, 2, 3}:
                return list(range(0, 64)) + list(range(128, 128 + 64))
            elif self.device_idx in {4, 5, 6, 7}:
                return list(range(64, 128)) + list(range(64 + 128, 128 + 128))
            else:
                raise RuntimeError('Unknown device_idx')
        elif scope == gpu_affinity.Scope.NODE:
            if self.device_idx in {0, 1}:
                return list(range(48, 64)) + list(range(176, 192))
            elif self.device_idx in {2, 3}:
                return list(range(16, 32)) + list(range(144, 160))
            elif self.device_idx in {4, 5}:
                return list(range(112, 128)) + list(range(240, 256))
            elif self.device_idx in {6, 7}:
                return list(range(80, 96)) + list(range(208, 224))
            else:
                raise RuntimeError('Unknown device_idx')
        else:
            raise RuntimeError('Unknown scope')


class DGXH100Device:
    def __init__(self, device_idx):
        super().__init__()
        self.device_idx = device_idx

    def get_cpu_affinity(self, scope):
        if scope in {gpu_affinity.Scope.SOCKET, gpu_affinity.Scope.NODE}:
            if self.device_idx in {0, 1, 2, 3}:
                return list(range(0, 56)) + list(range(112, 168))
            elif self.device_idx in {4, 5, 6, 7}:
                return list(range(56, 112)) + list(range(168, 224))
            else:
                raise RuntimeError('Unknown device_idx')
        else:
            raise RuntimeError('Unknown scope')


def build_get_thread_siblings_list(system):
    def dgx1v():
        return [(i, i + 40) for i in range(40)]

    def dgx2h():
        return [(i, i + 48) for i in range(48)]

    def dgxa100():
        return [(i, i + 128) for i in range(128)]

    def dgxh100():
        return [(i, i + 112) for i in range(112)]

    if system == 'dgx1v':
        return dgx1v
    elif system == 'dgx2h':
        return dgx2h
    elif system == 'dgxa100':
        return dgxa100
    elif system == 'dgxh100':
        return dgxh100
    else:
        raise RuntimeError


@unittest.mock.patch('pynvml.nvmlInit', new=lambda: None)
@unittest.mock.patch(
    'gpu_affinity.gpu_affinity.get_thread_siblings_list',
    new=build_get_thread_siblings_list('dgx1v'),
)
@unittest.mock.patch('gpu_affinity.gpu_affinity.Device', new=DGX1VDevice)
@unittest.mock.patch('os.sched_getaffinity', create=True, new=MockAffinity.get)
@unittest.mock.patch('os.sched_setaffinity', create=True, new=MockAffinity.set)
class DGX1V(unittest.TestCase):
    def setUp(self):
        self.num_cores = 80
        self.num_gpus = 8
        self.sibling_offset = 40

    def execute(
        self,
        mode_opt,
        scope_opt,
        balanced_opt,
        multithreading_opt,
        build_reference,
        min_physical_cores=1,
        max_physical_cores=None,
    ):
        for mode, scope, balanced, multithreading in itertools.product(
            mode_opt, scope_opt, balanced_opt, multithreading_opt
        ):
            offset = (
                self.sibling_offset
                if multithreading == gpu_affinity.Multithreading.ALL_LOGICAL
                else None
            )
            reference = build_reference(offset)
            params = {
                'mode': mode,
                'scope': scope,
                'balanced': balanced,
                'multithreading': multithreading,
                'min_physical_cores': min_physical_cores,
                'max_physical_cores': max_physical_cores,
            }
            with self.subTest(params=params):
                for gpu_id in range(self.num_gpus):
                    MockAffinity.reset(self.num_cores)
                    affinity = gpu_affinity.set_affinity(
                        gpu_id,
                        self.num_gpus,
                        **params,
                    )
                    self.assertEqual(reference[gpu_id], affinity)

    def test_all(self):
        mode_opt = [gpu_affinity.Mode.ALL]
        scope_opt = [gpu_affinity.Scope.NODE, gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(0, 20, offset=offset),
                1: cores_set(0, 20, offset=offset),
                2: cores_set(0, 20, offset=offset),
                3: cores_set(0, 20, offset=offset),
                4: cores_set(20, 40, offset=offset),
                5: cores_set(20, 40, offset=offset),
                6: cores_set(20, 40, offset=offset),
                7: cores_set(20, 40, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_single(self):
        mode_opt = [gpu_affinity.Mode.SINGLE]
        scope_opt = [gpu_affinity.Scope.NODE, gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(0, 1, offset=offset),
                1: cores_set(0, 1, offset=offset),
                2: cores_set(0, 1, offset=offset),
                3: cores_set(0, 1, offset=offset),
                4: cores_set(20, 21, offset=offset),
                5: cores_set(20, 21, offset=offset),
                6: cores_set(20, 21, offset=offset),
                7: cores_set(20, 21, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_single_unique(self):
        mode_opt = [gpu_affinity.Mode.SINGLE_UNIQUE]
        scope_opt = [gpu_affinity.Scope.NODE, gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(0, 1, offset=offset),
                1: cores_set(1, 2, offset=offset),
                2: cores_set(2, 3, offset=offset),
                3: cores_set(3, 4, offset=offset),
                4: cores_set(20, 21, offset=offset),
                5: cores_set(21, 22, offset=offset),
                6: cores_set(22, 23, offset=offset),
                7: cores_set(23, 24, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_unique_contiguous(self):
        mode_opt = [gpu_affinity.Mode.UNIQUE_CONTIGUOUS]
        scope_opt = [gpu_affinity.Scope.NODE, gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(0, 5, offset=offset),
                1: cores_set(5, 10, offset=offset),
                2: cores_set(10, 15, offset=offset),
                3: cores_set(15, 20, offset=offset),
                4: cores_set(20, 25, offset=offset),
                5: cores_set(25, 30, offset=offset),
                6: cores_set(30, 35, offset=offset),
                7: cores_set(35, 40, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_unique_interleaved(self):
        mode_opt = [gpu_affinity.Mode.UNIQUE_INTERLEAVED]
        scope_opt = [gpu_affinity.Scope.NODE, gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(0, 20, 4, offset=offset),
                1: cores_set(1, 20, 4, offset=offset),
                2: cores_set(2, 20, 4, offset=offset),
                3: cores_set(3, 20, 4, offset=offset),
                4: cores_set(20, 40, 4, offset=offset),
                5: cores_set(21, 40, 4, offset=offset),
                6: cores_set(22, 40, 4, offset=offset),
                7: cores_set(23, 40, 4, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )


@unittest.mock.patch('pynvml.nvmlInit', new=lambda: None)
@unittest.mock.patch(
    'gpu_affinity.gpu_affinity.get_thread_siblings_list',
    new=build_get_thread_siblings_list('dgx2h'),
)
@unittest.mock.patch('gpu_affinity.gpu_affinity.Device', new=DGX2HDevice)
@unittest.mock.patch('os.sched_getaffinity', create=True, new=MockAffinity.get)
@unittest.mock.patch('os.sched_setaffinity', create=True, new=MockAffinity.set)
class DGX2H(unittest.TestCase):
    def setUp(self):
        self.num_cores = 96
        self.num_gpus = 16
        self.sibling_offset = 48

    def execute(
        self,
        mode_opt,
        scope_opt,
        balanced_opt,
        multithreading_opt,
        build_reference,
        min_physical_cores=1,
        max_physical_cores=None,
    ):
        for mode, scope, balanced, multithreading in itertools.product(
            mode_opt, scope_opt, balanced_opt, multithreading_opt
        ):
            offset = (
                self.sibling_offset
                if multithreading == gpu_affinity.Multithreading.ALL_LOGICAL
                else None
            )
            reference = build_reference(offset)
            params = {
                'mode': mode,
                'scope': scope,
                'balanced': balanced,
                'multithreading': multithreading,
                'min_physical_cores': min_physical_cores,
                'max_physical_cores': max_physical_cores,
            }
            with self.subTest(params=params):
                for gpu_id in range(self.num_gpus):
                    MockAffinity.reset(self.num_cores)
                    affinity = gpu_affinity.set_affinity(
                        gpu_id,
                        self.num_gpus,
                        **params,
                    )
                    self.assertEqual(reference[gpu_id], affinity)

    def test_all(self):
        mode_opt = [gpu_affinity.Mode.ALL]
        scope_opt = [gpu_affinity.Scope.NODE, gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(0, 24, offset=offset),
                1: cores_set(0, 24, offset=offset),
                2: cores_set(0, 24, offset=offset),
                3: cores_set(0, 24, offset=offset),
                4: cores_set(0, 24, offset=offset),
                5: cores_set(0, 24, offset=offset),
                6: cores_set(0, 24, offset=offset),
                7: cores_set(0, 24, offset=offset),
                8: cores_set(24, 48, offset=offset),
                9: cores_set(24, 48, offset=offset),
                10: cores_set(24, 48, offset=offset),
                11: cores_set(24, 48, offset=offset),
                12: cores_set(24, 48, offset=offset),
                13: cores_set(24, 48, offset=offset),
                14: cores_set(24, 48, offset=offset),
                15: cores_set(24, 48, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_single(self):
        mode_opt = [gpu_affinity.Mode.SINGLE]
        scope_opt = [gpu_affinity.Scope.NODE, gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(0, 1, offset=offset),
                1: cores_set(0, 1, offset=offset),
                2: cores_set(0, 1, offset=offset),
                3: cores_set(0, 1, offset=offset),
                4: cores_set(0, 1, offset=offset),
                5: cores_set(0, 1, offset=offset),
                6: cores_set(0, 1, offset=offset),
                7: cores_set(0, 1, offset=offset),
                8: cores_set(24, 25, offset=offset),
                9: cores_set(24, 25, offset=offset),
                10: cores_set(24, 25, offset=offset),
                11: cores_set(24, 25, offset=offset),
                12: cores_set(24, 25, offset=offset),
                13: cores_set(24, 25, offset=offset),
                14: cores_set(24, 25, offset=offset),
                15: cores_set(24, 25, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_single_unique(self):
        mode_opt = [gpu_affinity.Mode.SINGLE_UNIQUE]
        scope_opt = [gpu_affinity.Scope.NODE, gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(0, 1, offset=offset),
                1: cores_set(1, 2, offset=offset),
                2: cores_set(2, 3, offset=offset),
                3: cores_set(3, 4, offset=offset),
                4: cores_set(4, 5, offset=offset),
                5: cores_set(5, 6, offset=offset),
                6: cores_set(6, 7, offset=offset),
                7: cores_set(7, 8, offset=offset),
                8: cores_set(24, 25, offset=offset),
                9: cores_set(25, 26, offset=offset),
                10: cores_set(26, 27, offset=offset),
                11: cores_set(27, 28, offset=offset),
                12: cores_set(28, 29, offset=offset),
                13: cores_set(29, 30, offset=offset),
                14: cores_set(30, 31, offset=offset),
                15: cores_set(31, 32, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_unique_contiguous(self):
        mode_opt = [gpu_affinity.Mode.UNIQUE_CONTIGUOUS]
        scope_opt = [gpu_affinity.Scope.NODE, gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(0, 3, offset=offset),
                1: cores_set(3, 6, offset=offset),
                2: cores_set(6, 9, offset=offset),
                3: cores_set(9, 12, offset=offset),
                4: cores_set(12, 15, offset=offset),
                5: cores_set(15, 18, offset=offset),
                6: cores_set(18, 21, offset=offset),
                7: cores_set(21, 24, offset=offset),
                8: cores_set(24, 27, offset=offset),
                9: cores_set(27, 30, offset=offset),
                10: cores_set(30, 33, offset=offset),
                11: cores_set(33, 36, offset=offset),
                12: cores_set(36, 39, offset=offset),
                13: cores_set(39, 42, offset=offset),
                14: cores_set(42, 45, offset=offset),
                15: cores_set(45, 48, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_unique_interleaved(self):
        mode_opt = [gpu_affinity.Mode.UNIQUE_INTERLEAVED]
        scope_opt = [gpu_affinity.Scope.NODE, gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(0, 24, 8, offset=offset),
                1: cores_set(1, 24, 8, offset=offset),
                2: cores_set(2, 24, 8, offset=offset),
                3: cores_set(3, 24, 8, offset=offset),
                4: cores_set(4, 24, 8, offset=offset),
                5: cores_set(5, 24, 8, offset=offset),
                6: cores_set(6, 24, 8, offset=offset),
                7: cores_set(7, 24, 8, offset=offset),
                8: cores_set(24, 48, 8, offset=offset),
                9: cores_set(25, 48, 8, offset=offset),
                10: cores_set(26, 48, 8, offset=offset),
                11: cores_set(27, 48, 8, offset=offset),
                12: cores_set(28, 48, 8, offset=offset),
                13: cores_set(29, 48, 8, offset=offset),
                14: cores_set(30, 48, 8, offset=offset),
                15: cores_set(31, 48, 8, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )


@unittest.mock.patch('pynvml.nvmlInit', new=lambda: None)
@unittest.mock.patch(
    'gpu_affinity.gpu_affinity.get_thread_siblings_list',
    new=build_get_thread_siblings_list('dgxa100'),
)
@unittest.mock.patch('gpu_affinity.gpu_affinity.Device', new=DGXA100Device)
@unittest.mock.patch('os.sched_getaffinity', create=True, new=MockAffinity.get)
@unittest.mock.patch('os.sched_setaffinity', create=True, new=MockAffinity.set)
class DGXA100(unittest.TestCase):
    def setUp(self):
        self.num_cores = 256
        self.num_gpus = 8
        self.sibling_offset = 128

    def execute(
        self,
        mode_opt,
        scope_opt,
        balanced_opt,
        multithreading_opt,
        build_reference,
        min_physical_cores=1,
        max_physical_cores=None,
    ):
        for mode, scope, balanced, multithreading in itertools.product(
            mode_opt, scope_opt, balanced_opt, multithreading_opt
        ):
            offset = (
                self.sibling_offset
                if multithreading == gpu_affinity.Multithreading.ALL_LOGICAL
                else None
            )
            reference = build_reference(offset)
            params = {
                'mode': mode,
                'scope': scope,
                'balanced': balanced,
                'multithreading': multithreading,
                'min_physical_cores': min_physical_cores,
                'max_physical_cores': max_physical_cores,
            }
            with self.subTest(params=params):
                for gpu_id in range(self.num_gpus):
                    MockAffinity.reset(self.num_cores)
                    affinity = gpu_affinity.set_affinity(
                        gpu_id,
                        self.num_gpus,
                        **params,
                    )
                    self.assertEqual(reference[gpu_id], affinity)

    def test_all_node(self):
        mode_opt = [gpu_affinity.Mode.ALL]
        scope_opt = [gpu_affinity.Scope.NODE]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(48, 64, offset=offset),
                1: cores_set(48, 64, offset=offset),
                2: cores_set(16, 32, offset=offset),
                3: cores_set(16, 32, offset=offset),
                4: cores_set(112, 128, offset=offset),
                5: cores_set(112, 128, offset=offset),
                6: cores_set(80, 96, offset=offset),
                7: cores_set(80, 96, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_all_socket(self):
        mode_opt = [gpu_affinity.Mode.ALL]
        scope_opt = [gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            socket0 = cores_set(0, 64, offset=offset)
            socket1 = cores_set(64, 128, offset=offset)
            reference = {
                0: socket0,
                1: socket0,
                2: socket0,
                3: socket0,
                4: socket1,
                5: socket1,
                6: socket1,
                7: socket1,
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_all_socket_max63(self):
        mode_opt = [gpu_affinity.Mode.ALL]
        scope_opt = [gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            socket0 = set.union(
                cores_set(0, 47, offset=offset),
                cores_set(48, 64, offset=offset),
            )
            socket1 = set.union(
                cores_set(64, 111, offset=offset),
                cores_set(112, 128, offset=offset),
            )
            reference = {
                0: socket0,
                1: socket0,
                2: socket0,
                3: socket0,
                4: socket1,
                5: socket1,
                6: socket1,
                7: socket1,
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
            max_physical_cores=63,
        )

    def test_single(self):
        mode_opt = [gpu_affinity.Mode.SINGLE]
        scope_opt = [gpu_affinity.Scope.NODE, gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(48, 49, offset=offset),
                1: cores_set(48, 49, offset=offset),
                2: cores_set(16, 17, offset=offset),
                3: cores_set(16, 17, offset=offset),
                4: cores_set(112, 113, offset=offset),
                5: cores_set(112, 113, offset=offset),
                6: cores_set(80, 81, offset=offset),
                7: cores_set(80, 81, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_single_unique(self):
        mode_opt = [gpu_affinity.Mode.SINGLE_UNIQUE]
        scope_opt = [gpu_affinity.Scope.NODE, gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(48, 49, offset=offset),
                1: cores_set(49, 50, offset=offset),
                2: cores_set(16, 17, offset=offset),
                3: cores_set(17, 18, offset=offset),
                4: cores_set(112, 113, offset=offset),
                5: cores_set(113, 114, offset=offset),
                6: cores_set(80, 81, offset=offset),
                7: cores_set(81, 82, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_unique_contiguous_node(self):
        mode_opt = [gpu_affinity.Mode.UNIQUE_CONTIGUOUS]
        scope_opt = [gpu_affinity.Scope.NODE]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(48, 56, offset=offset),
                1: cores_set(56, 64, offset=offset),
                2: cores_set(16, 24, offset=offset),
                3: cores_set(24, 32, offset=offset),
                4: cores_set(112, 120, offset=offset),
                5: cores_set(120, 128, offset=offset),
                6: cores_set(80, 88, offset=offset),
                7: cores_set(88, 96, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_unique_contiguous_socket(self):
        mode_opt = [gpu_affinity.Mode.UNIQUE_CONTIGUOUS]
        scope_opt = [gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: set.union(
                    cores_set(48, 56, offset=offset),
                    cores_set(32, 40, offset=offset),
                ),
                1: set.union(
                    cores_set(56, 64, offset=offset),
                    cores_set(40, 48, offset=offset),
                ),
                2: set.union(
                    cores_set(16, 24, offset=offset),
                    cores_set(0, 8, offset=offset),
                ),
                3: set.union(
                    cores_set(24, 32, offset=offset),
                    cores_set(8, 16, offset=offset),
                ),
                4: set.union(
                    cores_set(112, 120, offset=offset),
                    cores_set(96, 104, offset=offset),
                ),
                5: set.union(
                    cores_set(120, 128, offset=offset),
                    cores_set(104, 112, offset=offset),
                ),
                6: set.union(
                    cores_set(80, 88, offset=offset),
                    cores_set(64, 72, offset=offset),
                ),
                7: set.union(
                    cores_set(88, 96, offset=offset),
                    cores_set(72, 80, offset=offset),
                ),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_unique_interleaved_node(self):
        mode_opt = [gpu_affinity.Mode.UNIQUE_INTERLEAVED]
        scope_opt = [gpu_affinity.Scope.NODE]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(48, 64, 2, offset=offset),
                1: cores_set(49, 64, 2, offset=offset),
                2: cores_set(16, 32, 2, offset=offset),
                3: cores_set(17, 32, 2, offset=offset),
                4: cores_set(112, 128, 2, offset=offset),
                5: cores_set(113, 128, 2, offset=offset),
                6: cores_set(80, 96, 2, offset=offset),
                7: cores_set(81, 96, 2, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_unique_interleaved_socket(self):
        mode_opt = [gpu_affinity.Mode.UNIQUE_INTERLEAVED]
        scope_opt = [gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: set.union(
                    cores_set(48, 64, 2, offset=offset),
                    cores_set(32, 48, 2, offset=offset),
                ),
                1: set.union(
                    cores_set(49, 64, 2, offset=offset),
                    cores_set(33, 48, 2, offset=offset),
                ),
                2: set.union(
                    cores_set(16, 32, 2, offset=offset),
                    cores_set(0, 16, 2, offset=offset),
                ),
                3: set.union(
                    cores_set(17, 32, 2, offset=offset),
                    cores_set(1, 16, 2, offset=offset),
                ),
                4: set.union(
                    cores_set(112, 128, 2, offset=offset),
                    cores_set(96, 112, 2, offset=offset),
                ),
                5: set.union(
                    cores_set(113, 128, 2, offset=offset),
                    cores_set(97, 112, 2, offset=offset),
                ),
                6: set.union(
                    cores_set(80, 96, 2, offset=offset),
                    cores_set(64, 80, 2, offset=offset),
                ),
                7: set.union(
                    cores_set(81, 96, 2, offset=offset),
                    cores_set(65, 80, 2, offset=offset),
                ),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )


@unittest.mock.patch('pynvml.nvmlInit', new=lambda: None)
@unittest.mock.patch(
    'gpu_affinity.gpu_affinity.get_thread_siblings_list',
    new=build_get_thread_siblings_list('dgxh100'),
)
@unittest.mock.patch('gpu_affinity.gpu_affinity.Device', new=DGXH100Device)
@unittest.mock.patch('os.sched_getaffinity', create=True, new=MockAffinity.get)
@unittest.mock.patch('os.sched_setaffinity', create=True, new=MockAffinity.set)
class DGXH100(unittest.TestCase):
    def setUp(self):
        self.num_cores = 224
        self.num_gpus = 8
        self.sibling_offset = 112

    def execute(
        self,
        mode_opt,
        scope_opt,
        balanced_opt,
        multithreading_opt,
        build_reference,
        min_physical_cores=1,
        max_physical_cores=None,
    ):
        for mode, scope, balanced, multithreading in itertools.product(
            mode_opt, scope_opt, balanced_opt, multithreading_opt
        ):
            offset = (
                self.sibling_offset
                if multithreading == gpu_affinity.Multithreading.ALL_LOGICAL
                else None
            )
            reference = build_reference(offset)
            params = {
                'mode': mode,
                'scope': scope,
                'balanced': balanced,
                'multithreading': multithreading,
                'min_physical_cores': min_physical_cores,
                'max_physical_cores': max_physical_cores,
            }
            with self.subTest(params=params):
                for gpu_id in range(self.num_gpus):
                    MockAffinity.reset(self.num_cores)
                    affinity = gpu_affinity.set_affinity(
                        gpu_id,
                        self.num_gpus,
                        **params,
                    )
                    self.assertEqual(reference[gpu_id], affinity)

    def test_all(self):
        mode_opt = [gpu_affinity.Mode.ALL]
        scope_opt = [gpu_affinity.Scope.NODE, gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(0, 56, offset=offset),
                1: cores_set(0, 56, offset=offset),
                2: cores_set(0, 56, offset=offset),
                3: cores_set(0, 56, offset=offset),
                4: cores_set(56, 112, offset=offset),
                5: cores_set(56, 112, offset=offset),
                6: cores_set(56, 112, offset=offset),
                7: cores_set(56, 112, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_single(self):
        mode_opt = [gpu_affinity.Mode.SINGLE]
        scope_opt = [gpu_affinity.Scope.NODE, gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(0, 1, offset=offset),
                1: cores_set(0, 1, offset=offset),
                2: cores_set(0, 1, offset=offset),
                3: cores_set(0, 1, offset=offset),
                4: cores_set(56, 57, offset=offset),
                5: cores_set(56, 57, offset=offset),
                6: cores_set(56, 57, offset=offset),
                7: cores_set(56, 57, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_single_unique(self):
        mode_opt = [gpu_affinity.Mode.SINGLE_UNIQUE]
        scope_opt = [gpu_affinity.Scope.NODE, gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(0, 1, offset=offset),
                1: cores_set(1, 2, offset=offset),
                2: cores_set(2, 3, offset=offset),
                3: cores_set(3, 4, offset=offset),
                4: cores_set(56, 57, offset=offset),
                5: cores_set(57, 58, offset=offset),
                6: cores_set(58, 59, offset=offset),
                7: cores_set(59, 60, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_unique_contiguous(self):
        mode_opt = [gpu_affinity.Mode.UNIQUE_CONTIGUOUS]
        scope_opt = [gpu_affinity.Scope.NODE, gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(0, 14, offset=offset),
                1: cores_set(14, 28, offset=offset),
                2: cores_set(28, 42, offset=offset),
                3: cores_set(42, 56, offset=offset),
                4: cores_set(56, 70, offset=offset),
                5: cores_set(70, 84, offset=offset),
                6: cores_set(84, 98, offset=offset),
                7: cores_set(98, 112, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )

    def test_unique_interleaved(self):
        mode_opt = [gpu_affinity.Mode.UNIQUE_INTERLEAVED]
        scope_opt = [gpu_affinity.Scope.NODE, gpu_affinity.Scope.SOCKET]
        balanced_opt = [True, False]
        multithreading_opt = [
            gpu_affinity.Multithreading.ALL_LOGICAL,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        ]

        def build_reference(offset):
            reference = {
                0: cores_set(0, 56, 4, offset=offset),
                1: cores_set(1, 56, 4, offset=offset),
                2: cores_set(2, 56, 4, offset=offset),
                3: cores_set(3, 56, 4, offset=offset),
                4: cores_set(56, 112, 4, offset=offset),
                5: cores_set(57, 112, 4, offset=offset),
                6: cores_set(58, 112, 4, offset=offset),
                7: cores_set(59, 112, 4, offset=offset),
            }
            return reference

        self.execute(
            mode_opt,
            scope_opt,
            balanced_opt,
            multithreading_opt,
            build_reference,
        )
