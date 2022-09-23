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
from typing import Mapping, Optional, List, Dict

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
                 si_dict: Optional[Mapping[int, float]] = None,
                 **kwargs,
                 ):
        super().__init__(batch_size=batch_size, num_workers=num_workers, collate_fn=self._collate)
        self.energy_std = None
        self.si_tensor = self._create_si_tensor(si_dict)
        self.load_data(trip_file)

    def _create_si_tensor(self, si_dict: Mapping[int, float]):
        si_tensor = AtomicData.get_si_energies()
        if si_dict is not None:
            for key, value in si_dict.items():
                si_tensor[key -1] = value
        return si_tensor

    def load_data(self, trip_file: pathlib.Path):
        # Load data
        container = Container(trip_file)

        # Process data
        self.species_list = self._calc_species_list(container)
        self.energy_std = self._calc_energy_std(container)

        # Make dataset
        self.ds_train = self._make_ds(container, 'train')
        self.ds_val = self._make_ds(container, 'val')
        self.ds_test = self._make_ds(container, 'test')

    def _calc_energy_std(self, container):
        adjusted_energies_list = []
        species_data, _, energy_data, _, *_ = container.get_data('train')
        for species_tensor, energy_tensor in zip(species_data, energy_data):
            adjusted_energy_tensor = energy_tensor \
                - torch.sum(self.si_tensor[(species_tensor-1).tolist()])  # Subtract SI energies
            adjusted_energies_list.append(adjusted_energy_tensor)
        adjusted_energies = torch.cat(adjusted_energies_list)
        return torch.std(adjusted_energies)

    def _calc_species_list(self, container):
        species = set()
        for species_list in container.get_data('train')[0]:
            species = species.union(set(species_list.tolist()))
        return list(species)

    def _make_ds(self, container, name):
        ds = TrIPDataset(self.si_tensor,
                         self.energy_std,
                         *container.get_data(name),
                         )
        return ds

    def get_species_list(self):
        return self.species_list

    def get_energy_std(self):
        return self.energy_std

    @staticmethod
    def _collate(samples):
        species_list, pos_list, energy_list, forces_list, box_size_list = list(map(list, zip(*samples)))
        species = torch.cat(species_list)
        target = torch.stack(energy_list), torch.cat(forces_list)
        return species, pos_list, box_size_list, target

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
                pos_list.append(torch.tensor(pos_data[i][j], dtype=torch.float32))
                energy_list.append(torch.tensor(energy_data[i][j], dtype=torch.float32))
                forces_list.append(torch.tensor(forces_data[i][j], dtype=torch.float32))
                box_list.append(torch.tensor(box_data[i][j], dtype=torch.float32))
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
                 box_size_list: List[Tensor]):
        """
        :param si_tensor:           Tensor of self interaction energies sorted by atomic number.
        :param energy_std:          Standard deviation of energies adjusted by subtracting ebe's
        :param species_list:        List of tensors of atomic numbers (Shape: [N]).
        :param pos_list:            List of tensors of positions (Shape: [N,3]).
        :param energy_list:         List of floats of system energies.
        :param forces_list:         List of tensors of forces (Shape: [N,3]).
        :param box_size_list:       If box_size is not None then the graphs are constructed using periodic BC's using
                                    the values in box_size (Shape: [1] or [3])
        """

        self.si_tensor = si_tensor
        self.energy_std = energy_std
        self.species_list = species_list
        self.pos_list = pos_list
        self.energy_list = energy_list
        self.forces_list = forces_list
        self.box_size_list = box_size_list

        len_tensor = torch.tensor([len(pos_tensor) for pos_tensor in pos_list])
        self.cum_sum = [0] + torch.cumsum(len_tensor, dim=0).tolist()

    def __len__(self):
        return self.cum_sum[-1]

    def __getitem__(self, item):
        mol_idx = bisect.bisect_right(self.cum_sum, item) - 1  # Which molecules is being indexed
        conf_idx = item - self.cum_sum[mol_idx]  # Which conformation of the molecule is being indexed
        species = self.species_list[mol_idx]
        pos = self.pos_list[mol_idx][conf_idx]
        energy = self.energy_list[mol_idx][conf_idx]
        forces = self.forces_list[mol_idx][conf_idx]
        box_size = self.box_size_list[mol_idx][conf_idx]

        # Normalize dataset
        adjustment = torch.sum(self.si_tensor[(species-1).tolist()])
        energy = (energy-adjustment) / self.energy_std
        forces = forces / self.energy_std  # Linear property of derivatives requires this
        return species, pos, energy, forces, box_size
        