import argparse
import logging

from openmm import *
from openmm.app import *
from openmm.unit import *
import torch

from se3_transformer.runtime.utils import str2bool
from trip.data import AtomicData

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
        symbols = [atom.element.symbol for atom in topo.atoms()]
        species = get_species(topo)
        pos = pdbf.getPositions(asNumpy=True) / angstrom
        boxsize = topo.getUnitCellDimensions() / angstrom
        boxsize = torch.tensor(boxsize, dtype=torch.float, device=device)
    else:  # Water example
        species = [8, 1, 1]
        pos = torch.tensor([[0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0]], device=device, dtype=torch.float)
        boxsize = torch.tensor(float('inf'))
    masses_list = AtomicData.get_masses_list()
    masses_tensor = torch.tensor(masses_list)
    species_tensor = torch.tensor(species)
    m = masses_tensor[species_tensor-1].to(device)
    module = TrIPModule(species, **vars(args))

    # Minimation procedure
    if args.minimize:
        module.log_energy(pos, boxsize)
        logging.info('Beginning minimization')
        pos = module.minimize(pos, boxsize)
        module.log_energy(pos, boxsize)
        logging.info('Finished minimization!')
    
    # Frequency calculation
    m = m.repeat(3,1).T.flatten()  # Copy 3 times for 3Ds
    h = module.hess(pos, boxsize)
    F = h / torch.sqrt(m[:,None] * m[None,:])
    eigvals, eigvecs = torch.linalg.eig(F)
    freqs = torch.sqrt(eigvals).real.flatten()
    freqs, _ = torch.sort(freqs)
    logging.info(f'Frequencies (cm-1): {(2720.23*freqs[6:]).tolist()}')
