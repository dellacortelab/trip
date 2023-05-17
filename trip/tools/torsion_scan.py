import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

import torch

from trip.tools import get_species, TrIPModule
from trip.tools.constraints import DihedralConstraint

import MDAnalysis as mda
import networkx as nx


def parse_args():
    parser = argparse.ArgumentParser(description='run md simulations using TrIP')
    parser.add_argument('--pdb', type=str, help='Path to the directory with the input pdb file')
    parser.add_argument('--out', type=str, default='/results/',
                        help='The path to the output directory, default=/results/')
    parser.add_argument('--label', type=str, default='trip',
                        help='What to call the outputs')
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
    logging.info('|        Torsion Scan         |')
    logging.info('===============================')

    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(device)

    # Load data
    u = mda.Universe(args.pdb)
    u.atoms.guess_bonds()
    symbols = u.atoms.elements
    species = get_species(symbols)
    module = TrIPModule(species, **vars(args))
    boxsize = torch.full((3,), float('inf'), dtype=torch.float, device=device)

    # Select atoms
    atom_nums = [int(item)-1 for item in args.atom_nums.split(',')]
    dih = u.atoms[atom_nums].dihedral
    constraint = DihedralConstraint(atom_nums, 0, device, k=30)
    module.constraints.append(constraint)

    # Seperate molecule into two pieces
    g = nx.Graph()
    g.add_edges_from(u.atoms.bonds.to_indices())
    g.remove_edge(*atom_nums[1:3])
    a, b = (g.subgraph(c) for c in nx.connected_components(g))
    tail = a if atom_nums[2] in a.nodes() else b
    tail = u.atoms[tail.nodes()]
    bvec = dih[2].position - dih[1].position
    center = (tail & dih.atoms[1:3])[0]
    # Set dihederal to 0
    tail.rotateby(-dih.value(), bvec, point=center.position)
    u.atoms.write('/results/init.xyz') # Save for gaussian
    # Run calculations
    energies = np.empty(36)
    angles = np.empty_like(energies)
    for i in range(36):
        logging.info(f'Step {i}')
        module.constraints[0].equil = dih.value() * 3.14159 / 180
        pos = torch.tensor(u.atoms.positions, dtype=torch.float, device=device)
        for k in [1, 10, 100]:
             module.constraints[0].k = k
             pos = module.minimize(pos, boxsize)
        
        with torch.no_grad():
            angles[i] = module.constraints[0].calc_quantity(pos)
            energies[i] = module.forward(pos, boxsize, forces=False)
        tail.rotateby(10, bvec, point=center.position)
        
    base = os.path.join(args.out, args.label + '_torsion')
    np.savez(base + '.npz', angles=angles, energies=energies)

    plt.plot(angles, energies)
    plt.xlabel('Angle (rad)')
    plt.ylabel('Energy (Ha)')
    plt.savefig(base +  '.png', dpi=300, transparent=True)
    
    logging.info('Finished')
