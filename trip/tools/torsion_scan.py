import argparse
import logging
import numpy as np
import os

import MDAnalysis as mda
import networkx as nx

import torch

from trip.tools import TrIPModule
from trip.tools.constraints import DihedralConstraint
from trip.tools.utils import get_species



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


class TorsionScanner:
    def __init__(self, universe, atom_nums):
        # Create general system
        u = universe.copy()
        u.atoms.guess_bonds()
        symbols = u.atoms.elements
        self.species = get_species(symbols)  # This is for module

        # Create dihedral
        self.atom_nums = atom_nums
        dih = u.atoms[atom_nums].dihedral

        # Select part of molecule to rotate
        g = nx.Graph()
        g.add_edges_from(u.atoms.bonds.to_indices())
        g.remove_edge(*atom_nums[1:3])
        a, b = (g.subgraph(c) for c in nx.connected_components(g))
        tail = a if atom_nums[2] in a.nodes() else b
        tail = u.atoms[tail.nodes()]
        bvec = dih[2].position - dih[1].position
        center = (tail & dih.atoms[1:3])[0]

        # Save attributes
        self.bvec = bvec
        self.center = center
        self.dih = dih
        self.tail = tail
        self.u = u

    def min(self, module, angle=None, device='cuda'):
        module.species_tensor = torch.tensor(self.species, dtype=torch.long, device='cuda')
        if angle is not None:
            self.tail.rotateby(angle - self.dih.value(), self.bvec, point=self.center.position)
        module.constraints = [DihedralConstraint(self.atom_nums, self.dih.value() * 3.14159 / 180, device, 1)]
        pos = torch.tensor(self.u.atoms.positions, dtype=torch.float, device=device)
        boxsize = torch.full((3,), float('inf'), dtype=torch.float, device=device)
        for k in [1, 10, 100]:
             angle = module.constraints[0].k = k
             pos = module.minimize(pos, boxsize)
        
        with torch.no_grad():
            angle = module.constraints[0].calc_quantity(pos)
            energy = module.forward(pos, boxsize, forces=False)

        return pos, angle, energy

        

if __name__ == '__main__':
    # Setup
    args = parse_args()
    logging.getLogger().setLevel(logging.INFO)
    logging.info('============ TrIP =============')
    logging.info('|        Torsion Scan         |')
    logging.info('===============================')

    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(device)

    # Construct system
    universe = mda.Universe(args.pdb)
    atom_nums = [int(item)-1 for item in args.atom_nums.split(',')]
    torsion_scanner = TorsionScanner(universe, atom_nums)
    module = TrIPModule(torsion_scanner.species, **vars(args))

    # Run calculations
    energies = np.empty(36)
    angles = np.empty_like(energies)
    for i in range(36):
        logging.info(f'Step {i}')
        _, angle, energy = torsion_scanner.min(module, angle=10*i)
        angles[i] = angle
        energies[i] = energy
    
    # Save data
    base = os.path.join(args.out, args.label + '_torsion')
    np.savez(base + '.npz', angles=angles, energies=energies)

    #import matplotlib.pyplot as plt
    #plt.plot(angles, energies)
    #plt.xlabel('Angle (rad)')
    #plt.ylabel('Energy (Ha)')
    #plt.savefig(base +  '.png', dpi=300, transparent=True)
    
    logging.info('Finished')
