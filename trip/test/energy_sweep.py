from trip.data_loading import GraphConstructor
from trip.model import TrIP
from se3_transformer.runtime.utils import to_cuda
import matplotlib.pyplot as plt
import torch
import numpy as np

#initialize parameters

######Load Model############

model = TrIP.load('/results/model_ani1x.pth', map_location='cuda:0')
model.to('cuda:0')

class SE3Module(torch.nn.Module):
    def __init__(self, trained_model):
        super(SE3Module, self).__init__()
        self.model = trained_model
        self.species_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
        self.graph_constructor = GraphConstructor(trained_model.model.cutoff)

    def forward(self, species, positions, forces=True):
        species_tensor = torch.tensor([self.species_dict[atom] for atom in species], dtype=torch.int)
        species_tensor, positions = to_cuda([species_tensor, positions])
        graph = self.graph_constructor.create_graphs(positions, torch.tensor(float('inf'))) # Cutoff for 5-12 model is 3.0 A
        graph.ndata['species'] = species_tensor
        if forces:
            energy, forces = self.model(graph, forces=forces, create_graph=False)
            return energy.item(), forces
        else:
            energy = self.model(graph, forces=forces, create_graph=False)
            return energy.item()

symbols = ['H','C','N','O']
elements = ['Hydrogen','Carbon','Nitrogen','Oxygen']
for symbol, name in zip(symbols, elements):
    species = [symbol, symbol]
    sm = SE3Module(model)

    r_array = np.linspace(0,4.5,100)
    e_array = np.zeros_like(r_array)

    for i, r in enumerate(r_array):
        pos = torch.FloatTensor([[0,0,0], [r,0,0]])
        energy = sm(species, pos, forces=False)
        e_array[i] = float(energy)
    plt.plot(r_array, e_array, label = name)
    data = np.array([r_array, e_array])

plt.legend()
plt.ylim(-1, 1)
plt.savefig(f'/results/energy_sweep.png', dpi=300)
