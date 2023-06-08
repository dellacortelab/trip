# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import os
import h5py
import numpy as np
import torch

from trip.data_loading.container import Container


def iter_data_buckets(h5filename, keys=['wb97x_dz.energy']):
    """ Iterate over buckets of data in ANI HDF5 file.
    Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
    and other available properties specified by `keys` list, w/o NaN values.
    """
    keys = set(keys)
    keys.discard('atomic_numbers')
    keys.discard('coordinates')
    with h5py.File(h5filename, 'r') as f:
        for grp in f.values():
            Nc = grp['coordinates'].shape[0]
            mask = np.ones(Nc, dtype=bool)
            data = dict((k, grp[k][()]) for k in keys)
            for k in keys:
                v = data[k].reshape(Nc, -1)
                mask = mask & ~np.isnan(v).any(axis=1)
            if not np.sum(mask):
                continue
            d = dict((k, data[k][mask]) for k in keys)
            d['atomic_numbers'] = grp['atomic_numbers'][()]
            d['coordinates'] = grp['coordinates'][()][mask]
            yield d

species_data = []
pos_data = []
forces_data = []
energy_data = []
num_list = []

file_path = '/results/ani1xrelease.h5'
it = iter_data_buckets(file_path, keys=['wb97x_dz.forces', 'wb97x_dz.energy'])
for num, molecule in enumerate(it):
    species_data.append(torch.tensor(molecule['atomic_numbers'], dtype=torch.long))
    pos_data.append(torch.tensor(molecule['coordinates'], dtype=torch.float32))
    energy_data.append(torch.tensor(molecule['wb97x_dz.energy'], dtype=torch.float32))
    forces_data.append(torch.tensor(molecule['wb97x_dz.forces'], dtype=torch.float32))
    num_list.append(num)

num_array = np.arange(num)
train_idx = num_array[num_array % 20 != 0]
val_idx = num_array[num_array % 20 == 0]

container = Container()


def idx_lists(idx_list, *value_lists):
    new_value_lists = []
    for value_list in value_lists:
        new_value_lists.append([value_list[j] for j in idx_list])
    return new_value_lists

boxsize_data = [torch.full((pos_tensor.shape[0], 3), float('inf'), dtype=torch.float32) for pos_tensor in pos_data]

container.set_data('train', *idx_lists(train_idx, species_data, pos_data, energy_data, forces_data, boxsize_data))
container.set_data('val', *idx_lists(val_idx, species_data, pos_data, energy_data, forces_data, boxsize_data))


# Do testing 
species_data = []
pos_data = []
energy_data = []
forces_data = []

data_dir = '/results'
test_dir = os.path.join(data_dir, 'COMP6v1')
species_dict = {b'H': 1, b'C': 6, b'N': 7, b'O': 8}
for subdir in os.listdir(test_dir):
    subpath = os.path.join(test_dir, subdir)
    for file_name in os.listdir(subpath):
        filepath = os.path.join(subpath, file_name)
        with h5py.File(filepath, 'r') as f:
            for main in f.values():
                for mol in main.values():
                    species_data.append(torch.tensor([species_dict[atom] for atom in mol['species']], dtype=torch.long))
                    pos_data.append(torch.tensor(np.array(mol['coordinates']), dtype=torch.float32))
                    energy_data.append(torch.tensor(mol['energies'], dtype=torch.float32))
                    forces_data.append(-torch.tensor(np.array(mol['forces']), dtype=torch.float32))  # COMP6's forces have wrong sign


boxsize_data = [torch.full((pos_tensor.shape[0], 3), float('inf'), dtype=torch.float32) for pos_tensor in pos_data]

container.set_data('test', species_data, pos_data, energy_data, forces_data, boxsize_data)
save_path = os.path.join(data_dir, 'ani1x.h5')

container.save_data(save_path)

# Now create a small subset for testing the code
test_container = Container()

subsets = ['train', 'val', 'test']

for subset in subsets:
    data = container.get_data(subset)
    new_data = [data[0]]
    for i, category in enumerate(data[1:]):
        new_data.append([conf[0,None,...] for conf in category])
    test_container.set_data(subset, *new_data)

test_path = os.path.join(data_dir, 'test.h5')
test_container.save_data(test_path)