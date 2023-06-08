import matplotlib.pyplot as plt
import numpy as np
import colormaps as cmaps

# Read Gaussian log file
def read_log(fn='h2o.log'):
    with open(fn, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'Summary of the potential surface scan:' in line:
            lines = lines[i+3:]
            break
    for i, line in enumerate(lines):
        if '-----------' in line:
            lines = lines[:i]
            break
    a_array = np.empty(len(lines))
    b_array = np.empty_like(a_array)
    e_array = np.empty_like(a_array)

    for i, line in enumerate(lines):
        split = line.split()
        a_array[i] = float(split[1])
        b_array[i] = float(split[2])
        e_array[i] = float(split[3])

    shape = int(np.sqrt(i+1)), int(np.sqrt(i+1))
    a_array = a_array.reshape(*shape)
    b_array = b_array.reshape(*shape)
    e_array = e_array.reshape(*shape)
    e_array -= np.min(e_array)
    e_array *= 627.5
    return a_array, b_array, e_array

def read_npz(fn='trip_pes.npz'):
    data = np.load(fn)
    energies = data['energies']
    energies -= np.min(energies)
    energies*= 627.5
    return data['grid_a'], data['grid_b'], energies

# Read outputs and plot torsion scan data
fig, [ax1, ax2, ax3] = plt.subplots(1,3, figsize=(6.6,2.8))
plt.subplots_adjust(wspace=0.02, hspace=0.02)

a_array, b_array, e_array = read_log('/results/h2o.log')
options = {'cmap': cmaps.davos, 'levels': np.linspace(0, 600, 25)}
ax1.contourf(a_array, b_array, e_array, **options)
ax1.axis('square')
#ax1.set_xlabel(r'Bond length O-H$_1$ ($\AA$)')
ax1.set_ylabel(r'Bond length O-H$_2$ ($\AA$)')
ax1.set_title('DFT')

a_array, b_array, e_array = read_npz('/results/ani_pes.npz')
ax2.contourf(a_array, b_array, e_array, **options)
ax2.axis('square')
ax2.set_xlabel(r'Bond length O-H$_1$ ($\AA$)')
ax2.set_title('ANI-1x')
ax2.set_yticks([])

a_array, b_array, e_array = read_npz('/results/trip_pes.npz')
cp = ax3.contourf(a_array, b_array, e_array, **options)
ax3.axis('square')
#ax3.set_xlabel(r'Bond length O-H$_1$ ($\AA$)')
ax3.set_title('TrIP')
ax3.set_yticks([])

plt.savefig('/results/contours.png', dpi=300)

