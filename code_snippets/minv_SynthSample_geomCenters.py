import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------------
# DEFINE PARTICLES: compute particle centers from cuboid aggregations

data_dir = Path('deGroot2018_data/PDI-16803')

# Location and name of QDM and cuboid file
ScanFile = data_dir / 'Area1-90-fig2MMT.txt'
CuboidFile = data_dir / 'FWInput-FineCuboids-A1.txt'
# Load center and half lengths of cuboids making the grains (tomog data)
cuboid_data = np.loadtxt(CuboidFile, skiprows=0)
cuboid_data[:, 2] *= -1
cuboid_data_idxs = cuboid_data[:, 6].astype(np.int16)
cx, cy, cz, cdx, cdy, cdz = (cuboid_data[:, i] for i in range(6))
vols = 8 * cdx * cdy * cdz

# Compute centers of mass (geometric centre) per p. into the particles array
particles = np.zeros((len(np.unique(cuboid_data_idxs)), 4))
centre = np.zeros(3)
for i, particle_idx in enumerate(np.unique(cuboid_data_idxs)):

    p = cuboid_data_idxs == particle_idx
    particle_vol = vols[p].sum()
    centre[0] = np.sum(cx[p] * vols[p]) / particle_vol
    centre[1] = np.sum(cy[p] * vols[p]) / particle_vol
    centre[2] = np.sum(cz[p] * vols[p]) / particle_vol

    particles[i][:3] = centre
    particles[i][3] = particle_vol

np.savetxt('area1_ums_part_centers_vols.txt', particles)
