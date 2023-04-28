import torch
from torch import nn


class Constraint(nn.Module):
    def __init__(self, atom_nums, equil, device, k=1e3):
        super().__init__()
        self.atom_nums = torch.tensor(atom_nums, device=device)
        self.equil = torch.tensor(equil, device=device)
        self.k = torch.tensor(k, device=device)

    def calc_vec(self, pos, i, j):
        return pos[self.atom_nums[...,j]] - pos[self.atom_nums[...,i]]

    def forward(self, pos):
        quantity = self.calc_quantity(pos)
        diff = quantity - self.equil
        return torch.sum(self.error_fn(diff))
    
    def calc_quantity(self, pos):
        # Need to impliment for specific type of constraint
        pass

    def error_fn(self, diff):
        return self.k * diff**2


class DistanceConstraint(Constraint):
    def __init__(self, atom_nums, equil, device, k=1e3):
        super().__init__(atom_nums, equil, device, k)

    def calc_quantity(self, pos):
        vec = self.calc_vec(pos, 0, 1)
        return torch.norm(vec, dim=-1)
    

class AngleConstraint(Constraint):
    def __init__(self, atom_nums, equil, device, k=1e3):
        super().__init__(atom_nums, equil, device, k)

    def calc_quantity(self, pos):
        u = self.calc_vec(pos, 0, 1)
        v = self.calc_vec(pos, 1, 2)

        norm_u = torch.norm(u, dim=-1)
        norm_v = torch.norm(v, dim=-1)
        dot = torch.inner(u, v)
        return torch.arccos(dot / (norm_u * norm_v))


class DihederalConstraint(Constraint):
    def __init__(self, atom_nums, equil, device, k=1e3):
        super().__init__(atom_nums, equil, device, k)

    def calc_quantity(self, pos):
        ''' Praxeolitic formula '''
        # Calculate basic vectors
        u = self.calc_vec(pos, 1, 0)  # Note the order
        v = self.calc_vec(pos, 1, 2)
        w = self.calc_vec(pos, 2, 3)

        # Normalize bond vector
        v = v / torch.norm(v, dim=-1, keepdim=True)

        # Subtract vector rejections
        u = u - torch.inner(u, v) * v
        w = w - torch.inner(w, v) * v

        # Calculate and return angle
        x = torch.inner(u, w)
        y = torch.inner(torch.cross(u, v), w)
        return torch.atan2(y, x)

    def error_fn(self, diff):
        return 2 * self.k * (1 - torch.cos(diff))  # Periodic and approximates k * diff**2 when diff << 0
