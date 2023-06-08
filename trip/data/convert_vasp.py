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

import numpy as np
import torch

from trip.data_loading.container import Container
from trip.tools import get_species


with open('/results/compress/OUTCAR','r') as f:
    outfile = f.readlines()
    
start_idx = []
energy_idx = []
box_idx = []
for idx, line in enumerate(outfile):
    if ' POSITION                                       TOTAL-FORCE (eV/Angst)\n' in line:
        start_idx.append(idx)
        
    if 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)' in line:
        energy_idx.append(idx+2)
        
    if ' direct lattice vectors                 reciprocal lattice vectors' in line:
        box_idx.append(idx+1)

        positions = []
forces = []
for idx in start_idx:
    current_pos = []
    current_force = []
    for line in outfile[idx+2:idx+102]:
        data = np.array(line.split(),dtype=float)
        current_pos.append(data[:3])
        current_force.append(data[3:])
    positions.append(np.array(current_pos))
    forces.append(np.array(current_force))
pos_tensor = np.array(positions)
forces_tensor = np.array(forces)

energies = []
for idx in energy_idx:
    energies.append(float(outfile[idx].split()[-2]))
energy_tensor = np.array(energies)

box_tensor = []
for idx in box_idx:
    box = []
    for line in outfile[idx:idx+3]:
        box.append(np.array(line.split()[:3],dtype=float))
    box_tensor.append(np.array(box))
boxsize_tensor = np.array(box_tensor[1:])

with open('/results/compress/XDATCAR','r') as f:
    species_data = f.readlines()

atom_number = species_data[6].split()
atom_type = species_data[5].split()

species_tensor = []
for i in range(len(atom_number)):
    species_tensor += [atom_type[i]] * int(atom_number[i])
species_tensor = np.array(species_tensor)


# Change object type to tensor
species_tensor = torch.tensor(get_species(species_tensor), dtype=torch.long)
pos_tensor = torch.tensor(pos_tensor, dtype=torch.float32)
energy_tensor = torch.tensor(energy_tensor-EATOM, dtype=torch.float32) / 27.2107
forces_tensor = torch.tensor(forces_tensor, dtype=torch.float32) / 27.2107
boxsize_tensor = torch.tensor(boxsize_tensor, dtype=torch.float32)
boxsize_tensor = torch.diagonal(boxsize_tensor, dim1=1, dim2=2)

# Save in container
container = Container()

train_start = 100
val_start = 9500
test_start = 9900

container.set_data('train',
                   [species_tensor],
                   [pos_tensor[train_start:val_start]],
                   [energy_tensor[train_start:val_start]],
                   [forces_tensor[train_start:val_start]],
                   [boxsize_tensor[train_start:val_start]]
)

container.set_data('val',
                   [species_tensor],
                   [pos_tensor[val_start:test_start]],
                   [energy_tensor[val_start:test_start]],
                   [forces_tensor[val_start:test_start]],
                   [boxsize_tensor[val_start:test_start]]
)

container.set_data('test',
                   [species_tensor],
                   [pos_tensor[test_start:]],
                   [energy_tensor[test_start:]],
                   [forces_tensor[test_start:]],
                   [boxsize_tensor[test_start:]]
)

container.save_data('/results/vasp.h5')
