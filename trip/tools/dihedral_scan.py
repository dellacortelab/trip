import argparse
import logging
import matplotlib.pyplot as plt

from openmm import *
from openmm.app import *
from openmm.unit import *
import torch

from trip.tools import get_species, TrIPModule
from trip.tools.constraints import DihedralConstraint


def parse_args():
    parser = argparse.ArgumentParser(description='run md simulations using TrIP')
    parser.add_argument('--pdb', type=str, help='Path to the directory with the input pdb file')
    parser.add_argument('--out', type=str, default='/results/',
                        help='The path to the output directory, default=/results/')
    parser.add_argument('--model_file', type=str, default='/results/trip_vanilla.pth',
                        help='Path to model file, default=/results/trip_vanilla.pth')
    parser.add_argument('--atom_nums', type=str, help='Which atom numbers to run torsion scan over')
    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use, default=0')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    # Setup
    args = parse_args()
    logging.getLogger().setLevel(logging.INFO)

    logging.info('============ TrIP =============')
    logging.info('|        Dihederal Scan        |')
    logging.info('===============================')

    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(args.gpu)

    # Load data
    pdbf = PDBFile(args.pdb)
    topo = pdbf.topology
    symbols = [atom.element.symbol for atom in topo.atoms()]
    species = get_species(symbols)
    module = TrIPModule(species, **vars(args))
    pos = pdbf.getPositions(asNumpy=True) / angstrom
    pos = torch.tensor(pos, dtype=torch.float, device=device)
    boxsize = torch.full((3), float('inf'), dtype=torch.float, device=device)

    # Create constraint
    atom_nums = [int(item) for item in args.atom_nums.split(',')]
    constraint = DihedralConstraint(atom_nums, 0, device)
    module.constraints.append(constraint)

    angles = []
    energies = []

    for equil in torch.arange(0, 360, 36):
        constraint.equil = equil
        pos = module.minimize(pos, boxsize)
        angles.append(constraint.calc_quantity(pos))
        energies.append(module(pos, boxsize, forces=False))

    print(angles)
    print(energies)

    plt.plot(angles)
    plt.xlabel('Angle (rad)')
    plt.ylabel('Energy (Ha)')
    