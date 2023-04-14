from pathlib import Path
import mmt_dipole_cuboid_inversion as dci
import numpy as np

data_dir = Path('deGroot2018_data/PDI-16803')

# Location and name of QDM and cuboid file
ScanFile = data_dir / 'Area1-90-fig2MMT.txt'
CuboidFile = data_dir / 'FWInput-FineCuboids-A1.txt'

SQUID_sensor_domain = np.array([[0, 0], [350, 200]]) * 1e-6
SQUID_spacing = 1e-6
SQUID_deltax, SQUID_deltay, SQUID_area = 0.5e-6, 0.5e-6, 1e-12
SQUID_height = 2e-6

mag_inv = dci.DipoleCuboidInversion(
    None, SQUID_sensor_domain, SQUID_spacing,
    SQUID_deltax, SQUID_deltay, SQUID_area, SQUID_height)

mag_inv.read_files(ScanFile, CuboidFile, cuboid_scaling_factor=1e-6)
mag_inv.set_scan_domain(gen_sd_mesh_from='sensor_center_domain')

# We then compute the forward (Green's) matrix to be inverted
mag_inv.prepare_matrix(method='cython')

# And we do the inversion:
mag_inv.calculate_inverse(method='scipy_pinv', rtol=1e-25)
