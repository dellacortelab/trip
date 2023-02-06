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
from mendeleev.fetch import fetch_ionization_energies
from mendeleev import element

import torch


class AtomicData:
    NUM_ELEMENTS = 100 # Number of elements that this class supports
    @staticmethod
    def get_ebe_tensor(N=NUM_ELEMENTS):
        '''Get atomic electron binding energies'''
        ionization_energies = fetch_ionization_energies(degree=list(range(1,N+1))).head(N)  # Fetch ionization energies in eV
        ie_array = np.tril(ionization_energies.to_numpy())  # Convert from dataframe to numpy array
        ebe_energies = -np.sum(ie_array, axis=1)  # Add all ionization energies for each element to get self interation energy
        ebe_tensor = torch.from_numpy(ebe_energies)
        ebe_tensor /= 27.211_386_246  # Convert from eV to Ha
        return ebe_tensor

    @staticmethod
    def get_atomic_symbols_list(N=NUM_ELEMENTS):
        return [element(i).symbol for i in range(1,N+1)]

