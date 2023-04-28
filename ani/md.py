import argparse
import logging
from sys import stdout
import time

from openmm.app import *
from openmm import *
from openmm.unit import *

from openmmml import MLPotential

def announce(func):
    '''
    Prints when function starts and stops and runtime
    Uses code from: https://realpython.com/primer-on-python-decorators/
    '''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f'Starting {func.__name__!r}')
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logging.info(f'Finished {func.__name__!r} in {run_time:.4f} secs')
        return value
    return wrapper

@announce
def parse_args():
    parser = argparse.ArgumentParser(description='Start simulation')
    parser.add_argument('--pdb', help='The path to the receptor')
    parser.add_argument('--padding', help='Amount of padding (nm)', type=float, default=1.)
    parser.add_argument('--ionic_strength', help='Amount of ionic Strength (molar)', type=float, default=0.0)
    parser.add_argument('--cutoff', help='Nonbonded cutoff distance (nm)', type=float, default=1.)

    parser.add_argument('--pres', help='Barostat pressure (atm)', type=float, default=1.)
    parser.add_argument('--temp', help='System temperature (K)', type=float, default=300.)
    parser.add_argument('--freq', help='Barostat frequency', type=int, default=1)
    parser.add_argument('--fric', help='Friction in Langevin Integrator (1/ps)', type=float, default=1.)

    parser.add_argument('--dt', help='Step size (ps)', type=float, default=5e-4)
    parser.add_argument('--nvt', help='NVT equilibration length (ns)', type=float, default=0.1)
    parser.add_argument('--npt', help='NPT equilibration length (ns)', type=float, default=0.1)
    parser.add_argument('--prod', help='Production simulation length (ns)', type=int, default=int(1e4))

    parser.add_argument('--out', help='Directory to store output files', type=str, default='.')
    parser.add_argument('--gpu', type=str, help='the gpu id')
    args = parser.parse_args()
    return args

@announce
def load_modeller(pdb: str= None, **kwargs):
    pdb_file = PDBFile(pdb)
    modeller = Modeller(pdb_file.topology, pdb_file.positions)
    return modeller

@announce
def solvate(modeller, forcefield, padding: float = 1, ionic_strength: float=0.1, **kwargs):
    modeller.addSolvent(forcefield, padding=padding*nanometers, positiveIon='Na+', negativeIon='Cl-',
                        ionicStrength=ionic_strength*molar) #padding like Feig, but physiological salt concentration

@announce
def get_integrator(temp: float=300, fric: float=0.1, dt: float=2e-3, **kwargs):
    temperature = temp * kelvin
    friction = fric / picosecond
    step_size = dt * picoseconds
    integrator = openmm.LangevinIntegrator(temperature, friction, step_size)
    return integrator

@announce
def add_barostat(system, pres: float=1.0, temp: float=300., freq: float=1.0, **kwargs):
    pressure = pres * atmosphere  
    temperature = temp * kelvin
    barostat_frequency = freq
    barostat = openmm.MonteCarloBarostat(pressure, temperature, barostat_frequency)
    system.addForce(barostat)

@announce
def get_simulation(topology, positions, forcefield, cutoff=1.0, gpu: str='0', **kwargs):
    potential = MLPotential('ani2x')
    system = potential.createSystem(topology)
    integrator = get_integrator(**kwargs)
    platform = Platform.getPlatformByName('CUDA')
    properties = {'DeviceIndex': gpu, 'Precision': 'mixed'}
    simulation = Simulation(topology, system, integrator)#, platform=platform, platformProperties=properties)
    simulation.context.setPositions(positions)
    return simulation

@announce
def save_pdb(simulation, name: str='', out: str='.', **kwargs):
    position = simulation.context.getState(getPositions=True).getPositions()
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    path = os.path.join(out, f'{name}.pdb')
    pdbfile.PDBFile.writeFile(simulation.topology, position, open(path,'w'))

@announce
def minimize(simulation, **kwargs):
    simulation.minimizeEnergy()
    save_pdb(simulation, 'minimized', **kwargs)

@announce
def run_simulation(simulation, length, dt=2e-3, label='', **kwargs):
    simulation.step(int(1e3 * length / dt))
    save_pdb(simulation, label, **kwargs)

@announce
def add_reporters(simulation, dt: float=2e-3, out: str='.', **kwargs):
    simulation.reporters.append(StateDataReporter(stdout, int(1/(dt*10**-3)), step=True))
    simulation.reporters.append(DCDReporter(os.path.join(out,'trajectory.dcd'), int(0.05/(dt*10**-3)))) #write out DCD every 50 ps.



if __name__ == "__main__":
    # Logging setup
    logging.getLogger().setLevel(logging.INFO)
    logging.info('======================')
    logging.info('|    MD Simulation   |')
    logging.info('======================')

    # Load data, setup simulation
    args = parse_args()
    modeller = load_modeller(**vars(args))
    forcefield = ForceField('amber14-all.xml','amber14/tip3pfb.xml')
    #modeller.addHydrogens(forcefield)
    #solvate(modeller, forcefield, **vars(args))
    simulation = get_simulation(modeller.topology,
                                modeller.positions,
                                forcefield,
                                **vars(args))

    '''
    # Minimization
    minimize(simulation, **vars(args))

    # Equilibration
    logging.info('NVT equilibration')
    simulation.context.setVelocitiesToTemperature(args.temp * kelvin)
    run_simulation(simulation, length=args.nvt, label='nvt', **vars(args))
    
    logging.info('NPT equilibration')
    add_barostat(simulation.system, **vars(args))
    run_simulation(simulation, length=args.npt, label='npt', **vars(args))
    '''

    # Production run
    logging.info('Production Run')
    
    add_reporters(simulation, **vars(args))
    run_simulation(simulation, length=args.prod, label='final', **vars(args))

    logging.info('Finished MD Simulation!')
