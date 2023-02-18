from scipy.optimize import minimize
import torch
from torch.autograd.functional import hessian

from trip.data import AtomicData
from trip.data_loading import GraphConstructor

from trip.model import TrIP


class TrIPModule(torch.nn.Module):
    def __init__(self, topo, model_file, gpu, **vars):
        super().__init__() 
        self.device = f'cuda:{gpu}'
        self.model = TrIP.load(model_file, map_location=self.device)
        self.graph_constructor = GraphConstructor(cutoff=self.model.cutoff)

        symbol_list = AtomicData.get_atomic_symbols_list()
        symbol_dict = {symbol: i+1 for i, symbol in enumerate(symbol_list)}
        species = [atom.element.symbol for atom in topo.atoms()]
        self.species_tensor = torch.tensor([symbol_dict[atom] for atom in species],
                                            dtype=torch.int, device=self.device)
	
    def forward(self, pos, box_size, forces=True):
        graph = self.graph_constructor.create_graphs(pos, box_size) # Cutoff for 5-12 model is 3.0 A
        graph.ndata['species'] = self.species_tensor

        if forces:
            energy, forces = self.model(graph, forces=forces)
            return energy.item(), forces
        else:
            energy = self.model(graph, forces=forces)
            return energy.item()
        
    def energy_np(self, pos, box_size):
        pos = torch.tensor(pos, dtype=torch.float, device=self.device).reshape(-1,3)
        with torch.no_grad():
            energy = self.forward(pos, box_size=box_size, forces=False)
        return energy

    def jac_np(self, pos, box_size):
        pos = torch.tensor(pos, dtype=torch.float, device=self.device).reshape(-1,3)
        _, forces = self.forward(pos, box_size=box_size)
        return -forces.detach().cpu().numpy().flatten()

    def hess_np(self, pos, box_size):
        def energy(pos):
            graph = self.graph_constructor.create_graphs(pos, box_size)
            graph.ndata['species'] = self.species_tensor
            return self.model(graph, forces=False)
        pos = torch.tensor(pos, dtype=torch.float, device=self.device).reshape(-1,3)
        return hessian(energy, pos)
    
    def minimize(self, pos, box_size, method='CG'):
        res = minimize(self.energy_np, pos.cpu().numpy().flatten(), args=box_size, method=method,
                       jac=self.jac_np, hess=self.hess_np)
        pos = torch.tensor(res.x, dtype=torch.float, device=self.device).reshape(-1,3)
