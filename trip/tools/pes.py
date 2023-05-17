import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

from openmm import *
from openmm.app import *
from openmm.unit import *
import torch

import torchani

from trip.data_loading import GraphConstructor
from trip.model import TrIP


def parse_args():
    parser = argparse.ArgumentParser(description='run md simulations using TrIP')
    parser.add_argument('--out', type=str, default='/results/',
                        help='The path to the output directory, default=/results/')
    parser.add_argument('--label', type=str, default='trip',
                        help='What to call the outputs')
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

    # Prepare atomistic system 
    box_size = torch.tensor(float('inf'), dtype=torch.float, device=device)
    dists = torch.linspace(0.5, 4.5, 100, device=device)
    grid_a, grid_b = torch.meshgrid(dists, dists)
    pos = torch.zeros(len(dists), len(dists), 3, 3, device=device)
    ang = torch.tensor(104.5 * 3.141592 / 180) # 104.5 deg in rad
    pos[:,:,1,0] = grid_a  # Set X coordinate of first H
    pos[:,:,2,0] = grid_b * torch.cos(ang)  # Set X coordinate of second H
    pos[:,:,2,1] = grid_b * torch.sin(ang)  # Set Y coordinate of second H
    pos = pos.reshape(-1, 3, 3)
    species = torch.tensor([8, 1, 1], dtype=torch.long, device=device) 

    # Run Calculations
    if args.model_file == 'ani1x':
        model = torchani.models.ANI1x(periodic_table_index=True).to(device)
        species = species.unsqueeze(0).repeat([len(pos),1])
        with torch.no_grad():
            energies = model((species, pos)).energies
    else:
        model = TrIP.load(args.model_file, map_location=device)
        pos_list = list(torch.unbind(pos, dim=0))
        species_tensor = species.repeat(len(pos_list))
        box_size_list = len(pos_list) * [box_size]
        # Pass through model
        graph_constructor = GraphConstructor(cutoff=model.cutoff)
        graph = graph_constructor.create_graphs(pos_list, box_size_list)
        graph.ndata['species'] = species_tensor
        with torch.no_grad():
            energies = model(graph, forces=False)
    energies = energies.reshape(grid_a.shape)

    # Put everything on numpy
    to_np = lambda t: t.detach().cpu().numpy()
    grid_a = to_np(grid_a)
    grid_b = to_np(grid_b)
    energies = to_np(energies)

    # Save data
    base = os.path.join(args.out, args.label + '_pes')
    np.savez(base + '.npz', grid_a, grid_b, energies)

    fig, ax = plt.subplots(1,1)
    cp = ax.contourf(grid_a, grid_b, energies, cmap=cmaps.davos, levels=15)
    fig.colorbar(cp)
    ax.set_xlabel(r'Bond length O-H$_1$ ($\AA$)')
    ax.set_ylabel(r'Bond length O-H$_2$ ($\AA$)')
    ax.set_title(args.label)

    plt.savefig(base + '.png', dpi=300)
    
    logging.info('Finished')
