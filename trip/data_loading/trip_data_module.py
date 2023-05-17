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
# SPDX-License-Identifier: MIT

import bisect
import pathlib
from typing import List
from collections import Counter
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

from se3_transformer.data_loading.data_module import DataModule

from trip.data import AtomicData
from trip.data_loading import Container


class TrIPDataModule(DataModule):
    def __init__(self,
                 trip_file: pathlib.Path,
                 batch_size: int = 1,
                 num_workers: int = 8,
                 ebe_dict: dict = {},
                 **kwargs,
                 ):
        super().__init__(batch_size=batch_size, num_workers=num_workers, collate_fn=self._collate)
        self._load_data(trip_file, ebe_dict)

    def _load_data(self, trip_file: pathlib.Path, ebe_dict: dict):
        # Load data
        container = Container(trip_file)

        # Preprocess data
        self._species_list = self._calc_species_list(container)
        self._ebe_tensor = AtomicData.get_ebe_tensor()
        for s, e in ebe_dict.items():  # Use custom values
            self._ebe_tensor[s - 1] = e
        self._si_tensor, self._energy_std = self._calc_si(container)
        self._adjusted_ae_tensor = (self._ebe_tensor - self._si_tensor) / self._energy_std  # TODO: Check this is the correct value

        # Make dataset
        self.ds_train = self._make_ds(container, 'train')
        self.ds_val = self._make_ds(container, 'val')
        self.ds_test = self._make_ds(container, 'test')

    def _calc_si(self, container):
        species_data, _, energy_data, _, _ = container.get_data('train')
        species_list = self._species_list
        
        # Count number of species for each species_list in species_data
        A = [list() for _ in species_list]
        for species_tensor, energy_tensor in zip(species_data, energy_data):
            counts = Counter(species_tensor.tolist())
            num_copies = len(energy_tensor)
            for i, s in enumerate(species_list):
                A[i].extend(num_copies * [counts[s]])
        
        # Solve least squares problem
        A = np.array(A, dtype=float).T
        b = torch.cat(energy_data).numpy()
        sol = np.linalg.lstsq(A, b, rcond=None)

        # Create si_tensor
        si_list = sol[0].tolist()
        si_tensor = self._ebe_tensor.clone()  # Won't actually use ebe values
        for s, e in zip(species_list, si_list):  # Use linear regression values
            si_tensor[s - 1] = e  # -1 so H starts at 0.

        # Use residuals to calculate dataset std
        sum_sq_residuals = sol[1][0]
        std = np.sqrt(sum_sq_residuals / len(b))
        return si_tensor, std
                
    def _calc_species_list(self, container):
        species_data = container.get_data('train')[0]
        species = {s for species_tensor in species_data for s in species_tensor.tolist()}
        species_list = list(species)
        species_list.sort()
        return species_list

    def _make_ds(self, container, name):
        ds = TrIPDataset(self._si_tensor,
                         self._energy_std,
                         *container.get_data(name))
        return ds

    def add_atom_data(self, species_tensor, pos_list, energy_tensor, forces_tensor, boxsize_tensor):
        atom_species = torch.tensor(self._species_list)
        num_atoms = len(atom_species)
        species_tensor = torch.cat((species_tensor, atom_species))
        pos_list.extend(num_atoms*[torch.zeros((1,3), dtype=torch.float)])
        energy_tensor = torch.cat((energy_tensor, self._adjusted_ae_tensor[atom_species-1]))
        forces_tensor = torch.cat((forces_tensor, torch.zeros((num_atoms,3), dtype=torch.float)))
        boxsize_tensor = torch.cat((boxsize_tensor, torch.full((num_atoms,3), float('inf'), dtype=torch.float)))
        return (species_tensor, pos_list, energy_tensor, forces_tensor, boxsize_tensor), num_atoms

    @property
    def species_list(self):
        return self._species_list.copy()

    @property
    def si_tensor(self):
        return self._si_tensor.clone()

    @property
    def energy_std(self):
        return self._energy_std

    @staticmethod
    def _collate(samples):
        species_list, pos_list, energy_list, forces_list, boxsize_list = list(map(list, zip(*samples)))
        species_tensor = torch.cat(species_list)
        energy_tensor = torch.stack(energy_list)
        forces_tensor = torch.cat(forces_list)
        boxsize_tensor = torch.stack(boxsize_list, dim=0)
        return species_tensor, pos_list, energy_tensor, forces_tensor, boxsize_tensor

    @staticmethod
    def _to_ds_format(species_data, pos_data, energy_data, forces_data, box_data):
        species_list = []
        pos_list = []
        energy_list = []
        forces_list = []
        box_list = []
        for i in range(len(species_data)):
            for j in range(len(pos_data[i])):
                species_list.append(torch.tensor(species_data[i], dtype=torch.long))
                pos_list.append(torch.tensor(pos_data[i][j], dtype=torch.float))
                energy_list.append(torch.tensor(energy_data[i][j], dtype=torch.float))
                forces_list.append(torch.tensor(forces_data[i][j], dtype=torch.float))
                box_list.append(torch.tensor(box_data[i][j], dtype=torch.float))
        return species_list, pos_list, energy_list, forces_list, box_list

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("TrIP dataset")
        parser.add_argument('--trip_file', type=pathlib.Path, default=pathlib.Path('/results/ani1x.trip'),
                            help='Directory where the data is located or should be downloaded')
        return parent_parser


class TrIPDataset(Dataset):
    def __init__(self,
                 si_tensor: Tensor,
                 energy_std: float,
                 species_list: List[Tensor],
                 pos_list: List[Tensor],
                 energy_list: List[Tensor],
                 forces_list: List[Tensor],
                 boxsize_list: List[Tensor]):
        """
        :param si_tensor:           Tensor of self interaction energies sorted by atomic number.
        :param energy_std:          Standard deviation of energies adjusted by subtracting si energies.
        :param species_list:        List of tensors of atomic numbers (Shape: [N]).
        :param pos_list:            List of tensors of positions (Shape: [N,3]).
        :param energy_list:         List of floats of system energies.
        :param forces_list:         List of tensors of forces (Shape: [N,3]).
        :param boxsize_list:       If boxsize is not None then the graphs are constructed using periodic BC's using
                                    the values in boxsize (Shape: [1] or [3])
        """

        self.si_tensor = si_tensor
        self.energy_std = energy_std
        self.species_list = species_list
        self.pos_list = pos_list
        self.energy_list = energy_list
        self.forces_list = forces_list
        self.boxsize_list = boxsize_list

        len_tensor = torch.tensor([len(pos_tensor) for pos_tensor in pos_list])
        self.cumsum = [0] + torch.cumsum(len_tensor, dim=0).tolist()

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        mol_idx = bisect.bisect_right(self.cumsum, item) - 1  # Which molecules is being indexed
        conf_idx = item - self.cumsum[mol_idx]  # Which conformation of the molecule is being indexed
        species = self.species_list[mol_idx]
        pos = self.pos_list[mol_idx][conf_idx]
        energy = self.energy_list[mol_idx][conf_idx]
        forces = self.forces_list[mol_idx][conf_idx]
        boxsize = self.boxsize_list[mol_idx][conf_idx]
        
        # Normalize dataset
        adjustment = torch.sum(self.si_tensor[(species-1).tolist()])
        energy = (energy-adjustment) / self.energy_std
        forces = forces / self.energy_std  # Linear property of derivatives requires this
        return species, pos, energy, forces, boxsize
        