import argparse
import logging
import os
from sys import stdout

from openmm import *
from openmm.app import *
from openmm.unit import *
import torch

from se3_transformer.runtime.utils import str2bool

from trip.tools import TrIPModule


def parse_args():
    parser = argparse.ArgumentParser(description='run md simulations using TrIP')
    parser.add_argument('--pdb', type=str, help='Path to the directory with the input pdb file')
    parser.add_argument('--out', type=str, default='/results/',
                        help='The path to the output directory, default=/results/')
    parser.add_argument('--model_file', type=str, default='/results/trip_vanilla.pth',
                        help='Path to model file, default=/results/trip_vanilla.pth')
    parser.add_argument('--minimize', type=str2bool, nargs='?', const=True, default=True,
                        help='Whether to minimize the structure')
    parser.add_argument('--dt', type=float, default=0.5,
                        help='Step size in femtoseconds')
    parser.add_argument('--t', type=float, default=1.,
                        help='Simulation time in nanoseconds')
    parser.add_argument('--temp', type=float, default=298.,
                        help='Temperature in kelvin')
    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use, default=0')
    args = parser.parse_args()
    return args

def get_trip_force():
    trip_force = CustomExternalForce('c-fx*x-fy*y-fz*z')
    trip_force.addPerParticleParameter('c')  # Correction term to get correct energy
    trip_force.addPerParticleParameter('fx')
    trip_force.addPerParticleParameter('fy')
    trip_force.addPerParticleParameter('fz')
    return trip_force

def get_system(topo, trip_force):
    system = System()
    for atom in topo.atoms():
        system.addParticle(atom.element.mass)
    system.addForce(trip_force)
    for index, atom in enumerate(topo.atoms()):
        trip_force.addParticle(index, (0, 0, 0, 0) * kilocalorie_per_mole/angstrom)
    return system

def save_pdb(pos, name, out, **args):
    with open(os.path.join(out, name+'.pdb'), 'w') as f:
        pdbfile.PDBFile.writeFile(topo, pos*angstrom, f)
    
def get_simulation(topo, system, pos, temp, dt, out, **args):
    integrator = LangevinIntegrator(temp*kelvin, 1/picosecond, dt*femtosecond)
    simulation = Simulation(topo, system, integrator)
    simulation.context.setPositions(pos.tolist() * angstrom)
    simulation.context.setVelocitiesToTemperature(temp * kelvin)
    simulation.reporters.append(DCDReporter(os.path.join(out, 'trajectory.dcd'), 1))
    simulation.reporters.append(StateDataReporter(stdout, 1, step=True, temperature=True,
                                                  potentialEnergy=True, totalEnergy=True))
    return simulation

def log_energy(module, pos, box_size):
    energy = module(pos, box_size, forces=False)
    logging.info(f'Energy: {energy:.4f}')


if __name__ == '__main__':
    # Set up
    args = parse_args()
    logging.getLogger().setLevel(logging.INFO)

    logging.info('============ TrIP =============')
    logging.info('|      Molecular dynamics      |')
    logging.info('===============================')

    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(args.gpu)

    # Load data
    pdbf = PDBFile(args.pdb)
    topo = pdbf.topology
    module = TrIPModule(topo, **vars(args))
    pos = pdbf.getPositions(asNumpy=True) / angstrom
    pos = torch.tensor(pos, dtype=torch.float, device=device)
    box_size = topo.getUnitCellDimensions() / angstrom
    box_size = torch.tensor(box_size, dtype=torch.float, device=device)

    # Minimation procedure
    if args.minimize:
        logging.info('Beginning minimization')
        log_energy(module, pos, box_size)
        pos = module.minimize(pos, box_size)
        log_energy(module, pos, box_size)
        save_pdb(pos, 'minimized', **vars(args))
        logging.info('Finished minimization!')
    
    # Run simulation
    trip_force = get_trip_force()
    system = get_system(topo, trip_force)
    simulation = get_simulation(topo, system, pos, **vars(args))
    num_steps = int(1e6 * args.t / args.dt)  # 1e6 is the ratio of femtoseconds to nanoseconds

    logging.info('Beginning simulation')
    for i in range(num_steps):
        pos = simulation.context.getState(getPositions=True).getPositions()
        pos = torch.tensor([[p.x, p.y, p.z] for p in pos],
                            dtype=torch.float, device=device)*10.0 # Nanometer to Angstrom conversion
        
        energy, forces = module(pos, box_size)
        c = 627.5 * (torch.sum(pos @ forces) + energy).item() / len(pos) * kilocalorie_per_mole  # Energy correction per atom
        forces *= 627.5 * kilocalorie_per_mole / angstrom
        for index, atom in enumerate(topo.atoms()):
            trip_force.setParticleParameters(index, index, [c, *forces[index]])
            
        trip_force.updateParametersInContext(simulation.context)
        simulation.step(1)

    logging.info('Simulation finished successfully')
