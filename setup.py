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

import setuptools

README = (pathlib.Path(__file__).parent / 'README.md').read_text()

install_requires = [
    'pynvml==11.5.0',
]

setuptools.setup(
    name='gpu_affinity',
    version='0.1.0',
    author='NVIDIA Corporation',
    description='GPU Affinity',
    long_description=README,
    long_description_content_type='text/markdown',
    url='',
    packages=['gpu_affinity'],
    install_package_data=True,
    install_requires=install_requires,
    license='Apache2',
    license_file='./LICENSE',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Operating System :: POSIX :: Linux',
    ],
    python_requires='>=3.6',
    extras_require={
        'dev': [
            'black==23.10.1',
            'isort==5.10.1',
            'pre-commit==3.5.0',
        ],
    },
)
