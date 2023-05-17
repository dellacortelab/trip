import matplotlib.pyplot as plt
import numpy as np

# Read Gaussian log file
def read_log(fn='ephedrine.log'):
    with open(fn, 'r') as f:
        lines = f.readlines()
    line = ''.join(lines).replace('\n','').replace(' ','')
    data = line.split('\\HF=')[1].split('\\RMSD=')[0]
    return np.array([float(item) for item in data.split(',')]) * 627.5

# Read outputs from trip/tools/pes.py
def read_npz(fn='trip_torsion.npz'):
    data = np.load(fn)
    angles = data['angles']
    angles *= 180 / 3.141592
    angles[angles < 0] += 360
    energies = data['energies']
    idx = np.argsort(angles)
    return angles[idx], energies[idx]*627.5

plt.figure(figsize=(8,3))

energies = read_log('/results/ephedrine.log')
angles = 10 * np.arange(37)
plt.plot(angles, energies, label='DFT')

angles, energies = read_npz('/results/trip_torsion.npz')
plt.plot(angles, energies, 'or', label='TrIP')

angles, energies = read_npz('/results/ani_torsion.npz')
plt.plot(angles, energies, 'xg', label='ANI-1x')

plt.subplots_adjust(left=0.15, bottom=0.15)

plt.legend()
plt.title('Torsion Scans of Ephedrine')
plt.ylabel('Energy (kcal/mol)')
plt.xlabel('Torsion Angle (Deg)')

plt.ticklabel_format(useOffset=False)
plt.savefig('/results/torsion_scans.png', dpi=300, transparent=True)
