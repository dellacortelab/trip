# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT


import numpy as np
import torch
from mendeleev.fetch import fetch_ionization_energies
from mendeleev import element

class AtomicData:
    @staticmethod
    def get_si_energies(N=100):
        '''Get atomic self-interaction energies'''
        ionization_energies = fetch_ionization_energies(degree=list(range(1,N+1))).head(N)  # Fetch ionization energies in eV
        ie_array = np.tril(ionization_energies.to_numpy())  # Convert from dataframe to numpy array
        si_energies = np.sum(ie_array, axis=1)  # Add all ionization energies for each element to get self interation energy
        si_tensor = torch.tensor(si_energies, dtype=torch.float32)
        si_tensor /= 27.211_386_245_988_53  # Convert from eV to Ha: https://physics.nist.gov/cgi-bin/cuu/Value?hrev
        return si_tensor

    @staticmethod
    def get_atomic_symbols_list(N=100):
        return [element[i].symbol for i in range(1,N+1)]
