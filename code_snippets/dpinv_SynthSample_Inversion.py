from pathlib import Path
import mmt_dipole_cuboid_inversion as dci
import numpy as np

# Load data from the synthetic sample of de Groot et al. (2018)
data_dir = Path('deGroot2018_data/PDI-16803')

# Location and name of SQUID scan data, and tomog. cuboid file
ScanFile = data_dir / 'Area1-90-fig2MMT.txt'
CuboidFile = data_dir / 'FWInput-FineCuboids-A1.txt'
# Define sensor area using sensor center positions
SQUID_sensor_domain = np.array([[0, 0], [350, 200]]) * 1e-6
SQUID_spacing = 1e-6
SQUID_deltax, SQUID_deltay, SQUID_area = 0.5e-6, 0.5e-6, 1e-12
SQUID_height = 2e-6
# Use lower-left and upper-right sensors to define the scan domain
mag_inv = dci.DipoleCuboidInversion(
    None, SQUID_sensor_domain, SQUID_spacing,
    SQUID_deltax, SQUID_deltay, SQUID_area, SQUID_height)
# Read files and define scan area
mag_inv.read_files(ScanFile, CuboidFile, cuboid_scaling_factor=1e-6)
mag_inv.set_scan_domain(gen_sd_mesh_from='sensor_center_domain')
# We then compute the forward (Green's) matrix to be inverted
mag_inv.prepare_matrix(method='cython')
# And we do the inversion
mag_inv.calculate_inverse(method='scipy_pinv', rtol=1e-20)
