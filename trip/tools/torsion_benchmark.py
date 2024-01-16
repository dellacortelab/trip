import argparse
import logging
import numpy as np
import os
import pickle as pkl

import MDAnalysis as mda
from openff.qcsubmit.results import TorsionDriveResultCollection
from openff.qcsubmit.results.filters import ElementFilter
from openff.units import unit

import torch

from trip.tools import TrIPModule, TorsionScanner


def parse_args():
    parser = argparse.ArgumentParser(description='run md simulations using TrIP')
    parser.add_argument('--json', type=str, default='/results/2-0-0-td-set-v1.json',
                        help='Path to the directory with the input json file')
    parser.add_argument('--out', type=str, default='/results/',
                        help='The path to the output directory, default=/results/')
    parser.add_argument('--label', type=str, default='trip',
                        help='What to call the outputs')
    parser.add_argument('--model_file', type=str, default='/results/trip.pth',
                        help='Path to model file, default=/results/trip_vanilla.pth')
    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use, default=0')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=-1)
    
    args = parser.parse_args()
    return args

def construct_universe(molecule, conformer):
    species = [atom.symbol for atom in molecule.atoms]
    pos = conformer.to(unit.angstrom).magnitude
    tmp_fn = f'/tmp/{np.random.randint(int(1e4),int(1e5))}.xyz'
    with open(tmp_fn, 'w') as f:
        f.write(f'{len(species)}\n')
        f.write('Temporary file for MDA\n')
        for s, (x, y, z) in zip(species, pos):
            f.write(f'{s} {x:8.3f} {y:8.3f} {z:8.3f}\n')
    universe = mda.Universe(tmp_fn)
    return universe


if __name__ == '__main__':
    # Setup
    args = parse_args()
    logging.getLogger().setLevel(logging.INFO)
    logging.info('============ TrIP =============')
    logging.info('|     Torsion Benchmark       |')
    logging.info('===============================')
    
    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(device)

    module = TrIPModule(species=[], **vars(args))
    args.model_file = 'ani2x'
    module2 = TrIPModule(species=[], **vars(args))

    data = {'i': [],
            'j': [],
            'qc': {'pos': [],
                     'angle': [],
                     'energy': []},
            'trip': {'pos': [],
                       'angle': [],
                       'energy': []},
            'ani': {'pos': [],
                      'angle': [],
                      'energy': []}}

    # Run calculations
    element_filter = ElementFilter(allowed_elements=['H','C','N','O'])  # Filter out wrong atom types
    torsion_drive_result_collection = TorsionDriveResultCollection.parse_file(args.json)
    torsion_drive_result_collection = torsion_drive_result_collection.filter(element_filter)
    torsion_drive_records = torsion_drive_result_collection.to_records()
    for i, (torsion_drive_record, molecule) in enumerate(torsion_drive_records[args.start:args.stop]):
        for j, atom_nums in enumerate(torsion_drive_record.dict()['keywords']['dihedrals']):
            atom_nums = list(atom_nums)
            for grid_id, qc_conformer in zip(molecule.properties["grid_ids"], molecule.conformers):
                print(f'Running torsion scan {i} {j} {grid_id}')
                try:
                    universe = construct_universe(molecule, qc_conformer)
                    torsion_scanner = TorsionScanner(universe, atom_nums)
                    pos, angle, energy = torsion_scanner.min(module)
                    pos2, angle2, energy2 = torsion_scanner.min(module2)
                    data['i'].append(i)
                    data['j'].append(j)
                    data['qc']['pos'].append(qc_conformer.to(unit.angstrom).magnitude)
                    data['qc']['angle'].append(float(grid_id[1:-1]))
                    data['qc']['energy'].append(torsion_drive_record.dict()['final_energy_dict'][grid_id])
                    data['trip']['pos'].append(pos.detach().cpu().numpy())
                    data['trip']['angle'].append((angle*180/3.141592).detach().cpu().item())
                    data['trip']['energy'].append(energy)
                    data['ani']['pos'].append(pos2.detach().cpu().numpy())
                    data['ani']['angle'].append((angle2*180/3.141592).detach().cpu().item())
                    data['ani']['energy'].append(energy2)
                except:
                    print('error, continuing')


        with open(f'/results/torsion_benchmark_{args.gpu}.pkl', 'wb') as f:
            pkl.dump(data, f)
 
    logging.info('Finished')
