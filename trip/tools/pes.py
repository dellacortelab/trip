import argparse
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from openmm import *
from openmm.app import *
from openmm.unit import *
import torch

from trip.data_loading import GraphConstructor
from trip.model import TrIP


def parse_args():
    parser = argparse.ArgumentParser(description='run md simulations using TrIP')
    parser.add_argument('--out', type=str, default='/results/',
                        help='The path to the output directory, default=/results/')
    parser.add_argument('--model_file', type=str, default='/results/trip_vanilla.pth',
                        help='Path to model file, default=/results/trip_vanilla.pth')
    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use, default=0')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Setup
    args = parse_args()
    logging.getLogger().setLevel(logging.INFO)

    logging.info('============ TrIP =============')
    logging.info('|          H2O Scan           |')
    logging.info('===============================')

    device = f'cuda:{args.gpu}'
    torch.cuda.set_device(args.gpu)
    
    # Load data 
    model = TrIP.load(args.model_file, map_location=device)
    graph_constructor = GraphConstructor(cutoff=model.cutoff)
    species = torch.tensor([8, 1, 1], dtype=torch.int, device=device) - 1
    box_size = torch.tensor(float('inf'), dtype=torch.float, device=device)

    # Prepare positional data prelims
    lens = torch.linspace(0.5, 4.5, 81)
    grid_a, grid_b = torch.meshgrid(lens, lens)

    # Set positions
    pos = torch.zeros(len(angles), len(lens), 3, 3)
    ang = torch.tensor(104.5 *3.141592 / 180)
    pos[:,:,1,0] = grid_a  # Set X coordinate of first H
    pos[:,:,2,0] = grid_b * torch.cos(ang)  # Set X coordinate of second H
    pos[:,:,2,1] = grid_b * torch.sin(ang)  # Set Y coordinate of second H
    pos = pos.reshape(-1, 3, 3)

    # Put everything into the correct shape
    pos_list = torch.unbind(pos, dim=0)
    species_tensor = species.repeat(len(pos_list))
    box_size_list = len(pos_list) * [box_size]

    # Pass through model
    graph = graph_constructor.create_graphs(pos, box_size)
    graph.ndata['species'] = species_tensor
    with torch.no_grad():
        energies = model(graph, forces=False)
    energies = energies.reshape(grid_a.shape)

    # Plot 
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(grid_a, grid_b, energies)
    plt.show()

    fig, ax = plt.subplots(1,1)
    cp = ax.contourf(grid_a.detach().cpu(), grid_b.detach().cpu(), energies.detach().cpu())#, levels=torch.linspace(-75.4, -74.2, 25))
    fig.colorbar(cp)
    ax.set_xlabel(r'Bond length O-H$_1$ ($\AA$)')
    ax.set_ylabel(r'Bond length O-H$_2$ ($\AA$)')

    #ax.set_title('H2O Energy Scan')
    plt.savefig('trip_contour.png', dpi=300)