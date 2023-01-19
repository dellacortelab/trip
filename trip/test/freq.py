from scipy.optimize import minimize

from torch.autograd.functional import hessian

from trip.data_loading import GraphConstructor
from trip.model import TrIP
import torch

device = 'cuda:0'
model = TrIP.load('/results/model_ani1x.pth', map_location=device)

graph_constructor = GraphConstructor(model.cutoff)

# Positions of water
pos = torch.tensor([[0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0]], device=device, dtype=torch.float)
species = torch.tensor([8,1,1], device=device)
m = torch.tensor([16,1,1], device=device)

def energy_np(pos):
    pos = torch.tensor(pos, device=device, dtype=torch.float)
    return energy(pos).cpu().detach().numpy()

def energy(pos):
    graph = graph_constructor.create_graphs(pos.reshape(-1,3), torch.tensor(float('inf')))
    graph.ndata['species'] = species
    return model(graph, forces=False)

sol = minimize(energy_np, pos.flatten().cpu().numpy())
pos = torch.tensor(sol.x, dtype=torch.float, device=device)
print(f'Energy: {energy(pos).item()}')

h = hessian(energy, pos)
m = m.repeat(3,1).T.flatten()
F = h / torch.sqrt(m[:,None]*m[None,:])
eigvals, eigvecs = torch.linalg.eig(F)
freqs = torch.sqrt(eigvals).real.flatten()
freqs, order = torch.sort(freqs)
print(f'Frequencies (cm-1): {(2720.23*freqs[-3:]).tolist()}')
