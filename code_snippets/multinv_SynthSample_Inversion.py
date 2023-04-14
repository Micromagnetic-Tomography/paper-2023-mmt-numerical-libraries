import numpy as np
from pathlib import Path
import mmt_multipole_inversion as minv

# -----------------------------------------------------------------------------
# INVERSION

data_dir = Path('deGroot2018_data/PDI-16803')
SaveDir = Path('SyntheticSampleFiles')
SaveDir.mkdir(exist_ok=True)

# The area sensor formalism is implemented up to quadrupole order only
# The Area1_UMS_NPZ_ARRAYS.npz numpy file contains: grain centers and volumes
# The AREA1_UMS_METADICT.json contains: scanning surface properties
inv_area1_ums = minv.MultipoleInversion(
    SaveDir / "AREA1_UMS_METADICT.json",
    SaveDir / 'Area1_UMS_NPZ_ARRAYS.npz',
    expansion_limit='quadrupole',
    sus_functions_module='spherical_harmonics_basis_area')
# Load the scanning array manually:
inv_area1_ums.Bz_array = np.loadtxt(data_dir / 'Area1-90-fig2MMT.txt')
# Compute the inversion with a small relative tolerance
inv_area1_ums.compute_inversion(rcond=1e-30, method='sp_pinv')

# Compute magnetizations - every row has the magnetic moments of a single grain
# We get the first 3 columns for every row
mag_area1_ums = inv_area1_ums.inv_multipole_moments[:, :3]
mag_area1_ums /= inv_area1_ums.volumes[:, None]
mag_area1_ums = np.sqrt(np.sum(mag_area1_ums[:, :3] ** 2, axis=1))
