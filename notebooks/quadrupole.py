# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Imports

# %%
from mmt_multipole_inversion import MagneticSample
from mmt_multipole_inversion import MultipoleInversion
import numpy as np

# %% [markdown]
# # Quadrupole
#
# In the following piece of code we generate a perfect quadrupole from two dipoles, using the `MagneticSample` module. We also specify the dimensions of the measurement surface. We save the data in files with the `quadrupole_y-orientation` names.

# %%
# Scan height, Scan area x and y, sensor half-legth x and y (meter)
Hz, Sx, Sy, Sdx, Sdy = 2e-6, 20e-6, 20e-6, 0.1e-6, 0.1e-6
Lx, Ly, Lz = Sx * 0.9, Sy * 0.9, 5e-6  # Sample lengths (meter)

# Initialise class to create a sample (with user-defined or random particles)
sample = MagneticSample(Hz, Sx, Sy, Sdx, Sdy, Lx, Ly, Lz)

# Manually set the positions and dipole magnetic moments of two dipoles
Ms = 4.8e5
dipole_positions = np.array([[sample.Lx * 0.5 - 1e-6, sample.Ly * 0.5, -sample.Lz * 0.5],
                             [sample.Lx * 0.5 + 1e-6, sample.Ly * 0.5, -sample.Lz * 0.5]])
mu_s = Ms * (1 * 1e-18) * np.array([[0., 1., 0], [0., -1, 0]])
volumes = np.array([1e-18, 1e-18])
sample.generate_particles_from_array(dipole_positions, mu_s, volumes)

# Generate the dip field: the Bz field flux through the measurement surface
sample.generate_measurement_mesh()

# Redefine positions to make a single particle at the centre (ideal quadrupole)
sample.dipole_positions = np.array(
    [[sample.Lx * 0.5, sample.Ly * 0.5, -sample.Lz * 0.5]])
# Update the N of particles in the internal dict
sample.N_particles = 1

sample.save_data(filename='quadrupole_y-orientation')

# %% [markdown]
# Now we proceed to do the numerical inversions using two different basis for the multipole expansion. The fully orthogonal basis is the `spherical_harmonics_basis`, the other basis are for testing purposes.

# %%
# Inversions ------------------------------------------------------------------

shinv = MultipoleInversion('./MetaDict_quadrupole_y-orientation.json',
                           './MagneticSample_quadrupole_y-orientation.npz',
                           expansion_limit='quadrupole',
                           sus_functions_module='spherical_harmonics_basis')
shinv.generate_measurement_mesh()
shinv.generate_forward_matrix()
shinv.compute_inversion(method='sp_pinv')

mcinv = MultipoleInversion('./MetaDict_quadrupole_y-orientation.json',
                           './MagneticSample_quadrupole_y-orientation.npz',
                           expansion_limit='quadrupole',
                           sus_functions_module='maxwell_cartesian_polynomials')
mcinv.generate_measurement_mesh()
shinv.generate_forward_matrix()
mcinv.compute_inversion(method='sp_pinv')

# %% [markdown]
# Here the stray field signal matrix, the inverted scan signal matrix $\overline{\mathbf{B}}_{z}^{\text{inv}}$ and the residual~\cite{Zhdanov2015} matrix
#
# \begin{equation}
#     \overline{\mathbf{B}}_{z}^{\text{res}}=\overline{\mathbf{B}}_{z}^{\text{inv}} - \overline{\mathbf{B}}_{z}
# \end{equation}
#
# are computed. The residual error can be quantified by the relative error
#
# \begin{equation}
# B_{\text{err}}=\frac{\left\Vert \overline{\mathbf{B}}_{z}^{\text{res}} \right\Vert_{F}}{\left\Vert \overline{\mathbf{B}}_{z}\right\Vert_{F}},
# \end{equation}
#
# with $\left\Vert\cdot\right\Vert_{F}$ denoting the Frobenius norm.

# %%
sh_invBzNorm = np.linalg.norm(shinv.inv_Bz_array, ord='fro')
sh_BzNorm = np.linalg.norm(shinv.Bz_array, ord='fro')
sh_RelErrResidual = np.linalg.norm(shinv.inv_Bz_array - shinv.Bz_array, ord='fro') / sh_BzNorm

mc_invBzNorm = np.linalg.norm(mcinv.inv_Bz_array, ord='fro')
mc_BzNorm = np.linalg.norm(mcinv.Bz_array, ord='fro')
mc_RelErrResidual = np.linalg.norm(mcinv.inv_Bz_array - mcinv.Bz_array, ord='fro') / mc_BzNorm

# %%
print('Computing relative error of inversions Berr using Frobenius norm')

# %%
print(f'Berr using Spherical Harmonics basis: {sh_RelErrResidual:.15f}')

# %%
print(f'Berr using Maxwell-Cartesian polynomials: {mc_RelErrResidual:.15f}')

# %%
