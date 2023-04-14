# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import mmt_multipole_inversion as minv
from pathlib import Path

# %%
dataDir = Path("hawaiian_sample_data/")

# %%
list(dataDir.glob('*'))

# %%
cuboidFile = dataDir / "500_500_Cuboids.txt"
bzFile = dataDir / "500_500_Bzup_OrgQDM.txt"

# %%
a = np.loadtxt(bzFile)

# %%
a.shape

# %%
417 * 1.2

# %%
cuboidData = np.loadtxt(cuboidFile)
cuboidData[:, 2] *= -1.
cuboidData_idxs = cuboidData[:, -1].astype(np.int32)
cx, cy, cz, cdx, cdy, cdz = (cuboidData[:, i] for i in range(6))
vols = 8 * cdx * cdy * cdz

# %%
# Compute centre of mass (geometric centre)
particles = np.zeros((len(np.unique(cuboidData_idxs)), 4))
centre = np.zeros(3)
for i, particle_idx in enumerate(np.unique(cuboidData_idxs)):

    p = cuboidData_idxs == particle_idx
    particle_vol = vols[p].sum()
    centre[0] = np.sum(cx[p] * vols[p]) / particle_vol
    centre[1] = np.sum(cy[p] * vols[p]) / particle_vol
    centre[2] = np.sum(cz[p] * vols[p]) / particle_vol

    particles[i][:3] = centre
    particles[i][3] = particle_vol

# %%
# Scale the positions and columes by micrometres
np.savez(BASE_DIR / 'Area1_ARM_NPZ_ARRAYS', particle_positions=particles[:, :3] * 1e-6,
         volumes=particles[:, 3] * 1e-18)

# Set dictionary
metadict = {}
metadict["Scan height Hz"] = 
metadict["Scan area x-dimension Sx"] = 500.4 * 1e-6
metadict["Scan area y-dimension Sy"] = 500.4 * 1e-6
metadict["Scan x-step Sdx"] = 1.2e-6
metadict["Scan y-step Sdy"] = 1.2e-6
metadict["Time stamp"] = '0000'
metadict["Number of particles"] = particles.shape[0]
# The scan has origin (0, 0) by default thus this coordinate is used
# as the first measurement point
# metadict["Scan origin x"] = 0.0e-6
# metadict["Scan origin y"] = 0.0e-6

with open(BASE_DIR / "AREA1_ARM_METADICT.json", 'w') as f:
    json.dump(metadict, f)

# %%
inv_area1_arm_quad = minv.MultipoleInversion(BASE_DIR / "AREA1_ARM_METADICT.json",
                                             BASE_DIR / 'Area1_ARM_NPZ_ARRAYS.npz',
                                             expansion_limit='quadrupole',
                                             sus_functions_module='spherical_harmonics_basis'
                                             )

# %%
