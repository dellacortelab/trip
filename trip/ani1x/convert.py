import os
import h5py
import numpy as np
import torch

from trip.data_loading.trip_container import TrIPContainer


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
    species_data.append(torch.tensor(molecule['atomic_numbers'], dtype=torch.int))
    pos_data.append(torch.tensor(molecule['coordinates'], dtype=torch.float))
    energy_data.append(torch.tensor(molecule['wb97x_dz.energy'], dtype=torch.float))
    forces_data.append(torch.tensor(molecule['wb97x_dz.forces'], dtype=torch.float))
    num_list.append(num)

num_array = np.arange(num)
train_idx = num_array[num_array % 20 == 0]
val_idx = num_array[num_array %20 != 0]

container = TrIPContainer()


def idx_lists(idx_list, *value_lists):
    new_value_lists = []
    for value_list in value_lists:
        new_value_lists.append([value_list[j] for j in idx_list])
    return new_value_lists

box_size_data = [torch.full((pos_tensor.shape[0], 3), float('inf')) for pos_tensor in pos_data]

container.set_data('train', *idx_lists(train_idx, species_data, pos_data, energy_data, forces_data, box_size_data))
container.set_data('val', *idx_lists(val_idx, species_data, pos_data, energy_data, forces_data, box_size_data))


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
                    species_data.append(torch.tensor([species_dict[atom] for atom in mol['species']], dtype=torch.int))
                    pos_data.append(torch.tensor(np.array(mol['coordinates']), dtype=torch.float))
                    energy_data.append(torch.tensor(mol['energies'], dtype=torch.float))
                    forces_data.append(torch.tensor(np.array(mol['forces']), dtype=torch.float))


box_size_data = [torch.full((pos_tensor.shape[0], 3), float('inf')) for pos_tensor in pos_data]

container.set_data('test', species_data, pos_data, energy_data, forces_data, box_size_data)
save_path = os.path.join(data_dir, 'ani1x.trip')

container.save_data(save_path)
