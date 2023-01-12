from scipy.optimize import minimize

from torch.autograd.functional import hessian

from trip.data_loading import GraphConstructor
from trip.model import TrIP
import torch

device = 'cuda:0'
model = TrIP.load('/results/model_ani1x.pth', map_location=device)

graph_constructor = GraphConstructor(model.cutoff)

# Positions of water
pos = torch.tensor([[0.00000, 0.00000, 0.11779],
                    [0.00000, 0.75545, -0.47116],
                    [0.00000, -0.75545, -0.47116]], device=device)
species = torch.tensor([8,1,1], device=device)

def energy_np(pos):
    pos = torch.tensor(pos, device=device, dtype=torch.float).reshape(-1,3)
    return energy(pos).cpu().detach().numpy()

def energy(pos):
    graph = graph_constructor.create_graphs(pos, torch.tensor(float('inf')))
    graph.ndata['species'] = torch.tensor(species, device=device)
    return model(graph, forces=False)

sol = minimize(energy_np, pos.flatten().cpu().numpy())
pos = torch.tensor(sol.x, dtype=torch.float, device=device)
print(f'Energy: {energy(pos)}')
h = hessian(model, pos)
m = torch.tensor([16,16,16,1,1,1,1,1,1], device=device)
F = h / torch.sqrt(m[:,None]*m[None,:])
eigvals = torch.linalg.eig(F)
freqs = torch.sqrt(eigvals).real.flatten()
freqs = torch.sort(freqs)[0]
print(f'Frequencies (cm-1): {2720.23*freqs[-3:]}')
