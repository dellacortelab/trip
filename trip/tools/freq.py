import argparse
import logging

from openmm import *
from openmm.app import *
from openmm.unit import *
import torch

from se3_transformer.runtime.utils import str2bool
from trip.data import atomic_data

from trip.tools.md import get_species
from trip.tools.module import TrIPModule


def parse_args():
    parser = argparse.ArgumentParser(description='run md simulations using TrIP')
    parser.add_argument('--pdb', type=str, help='Path to the directory with the input pdb file', default='')
    parser.add_argument('--model_file', type=str, default='/results/trip_vanilla.pth',
                        help='Path to model file, default=/results/trip_vanilla.pth')
    parser.add_argument('--minimize', type=str2bool, nargs='?', const=True, default=True,
                        help='Whether to minimize the structure')
    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use, default=0')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Setup
    args = parse_args()
    logging.getLogger().setLevel(logging.INFO)

    logging.info('============ TrIP =============')
    logging.info('|      Frequency Analysis     |')
    logging.info('===============================')

    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(args.gpu)

    # Load data
    if len(args.pdb):
        pdbf = PDBFile(args.pdb)
        topo = pdbf.topology
        species = get_species(topo)
        pos = pdbf.getPositions(asNumpy=True) / angstrom
        box_size = topo.getUnitCellDimensions() / angstrom
        box_size = torch.tensor(box_size, dtype=torch.float, device=device)
    else:  # Water example
        species = [8, 1, 1]
        pos = torch.tensor([[0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0]], device=device, dtype=torch.float)
        box_size = torch.tensor(float('inf'))
    masses_list = atomic_data.get_atomic_masses_list()
    masses_tensor = torch.tensor(masses_list)
    m = masses_tensor[species]
    module = TrIPModule(species, **vars(args))

    # Minimation procedure
    if args.minimize:
        module.log_energy(pos, box_size)
        logging.info('Beginning minimization')
        pos = module.minimize(pos, box_size)
        module.log_energy(pos, box_size)
        logging.info('Finished minimization!')
    
    # Frequency calculation
    m = m.repeat(3,1).T.flatten()  # Copy 3 times for 3Ds
    h = module.hess(pos, box_size)
    F = h / torch.sqrt(m[:,None] * m[None,:])
    eigvals, eigvecs = torch.linalg.eig(F)
    freqs = torch.sqrt(eigvals).real.flatten()
    freqs, _ = torch.sort(freqs)
    logging.info(f'Frequencies (cm-1): {(2720.23*freqs[6:]).tolist()}')
