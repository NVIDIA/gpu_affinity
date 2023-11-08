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

import pathlib
import unittest
import unittest.mock

import gpu_affinity


def get_mock_open(files: dict[str, str]):
    def open_mock(filename, *args, **kwargs):
        for expected_filename, content in files.items():
            if filename == expected_filename:
                return unittest.mock.mock_open(read_data=content).return_value
        raise FileNotFoundError('(mock) Unable to open {filename}')

    return unittest.mock.MagicMock(side_effect=open_mock)


def thread_siblings_lists_fnames(N):
    return [
        f'/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list'
        for i in range(N)
    ]


class CheckSocketAffinities(unittest.TestCase):
    def test_disjoint(self):
        affinities = [[0, 1, 2, 3], [4, 5, 6, 7]]
        gpu_affinity.gpu_affinity.check_affinities(affinities)

    def test_identical(self):
        affinities = [[0, 1, 2, 3], [0, 1, 2, 3]]
        gpu_affinity.gpu_affinity.check_affinities(affinities)

    def test_overlap(self):
        affinities = [[0, 1, 2, 3], [1, 2, 3]]
        with self.assertRaises(gpu_affinity.GPUAffinityError):
            gpu_affinity.gpu_affinity.check_affinities(affinities)

    def test_single(self):
        affinities = [[0, 1, 2, 3]]
        gpu_affinity.gpu_affinity.check_affinities(affinities)


class GetThreadSiblingsList(unittest.TestCase):
    def test_missing(self):
        pass

    @unittest.mock.patch(
        'builtins.open',
        get_mock_open(
            dict(zip(thread_siblings_lists_fnames(2), ['0,1', '2,3']))
        ),
    )
    @unittest.mock.patch.object(
        pathlib.Path, 'glob', new=lambda x, y: thread_siblings_lists_fnames(2)
    )
    def test_mocked_simple(self):
        siblings_list = gpu_affinity.gpu_affinity.get_thread_siblings_list()
        self.assertEqual(siblings_list, [(0, 1), (2, 3)])

    @unittest.mock.patch(
        'builtins.open',
        get_mock_open(
            dict(zip(thread_siblings_lists_fnames(2), ['0,1', '0,1']))
        ),
    )
    @unittest.mock.patch.object(
        pathlib.Path, 'glob', new=lambda x, y: thread_siblings_lists_fnames(2)
    )
    def test_mocked_duplicates(self):
        siblings_list = gpu_affinity.gpu_affinity.get_thread_siblings_list()
        self.assertEqual(siblings_list, [(0, 1)])


class BuildThreadSiblingsDict(unittest.TestCase):
    def test_1_item(self):
        siblings_list = [(0, 1)]
        siblings_dict = gpu_affinity.gpu_affinity.build_thread_siblings_dict(
            siblings_list
        )
        self.assertEqual(siblings_dict, {0: (0, 1), 1: (0, 1)})

    def test_2_item(self):
        siblings_list = [(0, 1), (2, 3)]
        siblings_dict = gpu_affinity.gpu_affinity.build_thread_siblings_dict(
            siblings_list
        )
        self.assertEqual(
            siblings_dict, {0: (0, 1), 1: (0, 1), 2: (2, 3), 3: (2, 3)}
        )


class GroupListByDict(unittest.TestCase):
    def test_simple_1(self):
        siblings_dict = {0: (0, 1), 1: (0, 1)}
        siblings_key = lambda x: siblings_dict.get(x, (x,))
        affinity = [0, 1]
        grouped = gpu_affinity.gpu_affinity.group_list_by_key(
            affinity, siblings_key
        )
        self.assertEqual(grouped, [(0, 1)])

    def test_simple_2(self):
        siblings_dict = {0: (0, 1), 1: (0, 1), 2: (2, 3), 3: (2, 3)}
        siblings_key = lambda x: siblings_dict.get(x, (x,))
        affinity = [0, 1, 2, 3]
        grouped = gpu_affinity.gpu_affinity.group_list_by_key(
            affinity, siblings_key
        )
        self.assertEqual(grouped, [(0, 1), (2, 3)])

    def test_missing_dict(self):
        siblings_dict = {}
        siblings_key = lambda x: siblings_dict.get(x, (x,))
        affinity = [0, 1]
        grouped = gpu_affinity.gpu_affinity.group_list_by_key(
            affinity, siblings_key
        )
        self.assertEqual(grouped, [(0,), (1,)])


class UngroupAffinities(unittest.TestCase):
    def test_simple_all_logical(self):
        affinities = [[((0, 1),), ((2, 3),)]]
        ungrouped = gpu_affinity.gpu_affinity.ungroup_all_and_check_count(
            affinities,
            gpu_affinity.Scope.NODE,
            gpu_affinity.Multithreading.ALL_LOGICAL,
        )
        self.assertEqual(ungrouped, [[0, 1, 2, 3]])

    def test_simple_single_logical(self):
        affinities = [[((0, 1),), ((2, 3),)]]
        ungrouped = gpu_affinity.gpu_affinity.ungroup_all_and_check_count(
            affinities,
            gpu_affinity.Scope.SOCKET,
            gpu_affinity.Multithreading.SINGLE_LOGICAL,
        )
        self.assertEqual(ungrouped, [[0, 2]])

    def test_unknown_mode(self):
        affinities = [[((0, 1),), ((2, 3),)]]
        with self.assertRaises(gpu_affinity.GPUAffinityError):
            gpu_affinity.gpu_affinity.ungroup_all_and_check_count(
                affinities, 'node', 'does_not_exist'
            )
