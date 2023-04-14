from mmt_multipole_inversion import MagneticSample
from mmt_multipole_inversion import MultipoleInversion
import numpy as np

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

# Inversions ------------------------------------------------------------------

shinv = MultipoleInversion('./MetaDict_quadrupole_y-orientation.json',
                           './MagneticSample_quadrupole_y-orientation.npz',
                           expansion_limit='quadrupole',
                           sus_functions_module='spherical_harmonics_basis')
shinv.generate_measurement_mesh()
shinv.compute_inversion(method='sp_pinv2')

mcinv = MultipoleInversion('./MetaDict_quadrupole_y-orientation.json',
                           './MagneticSample_quadrupole_y-orientation.npz',
                           expansion_limit='quadrupole',
                           sus_functions_module='maxwell_cartesian_polynomials')
mcinv.generate_measurement_mesh()
mcinv.compute_inversion(method='sp_pinv2')