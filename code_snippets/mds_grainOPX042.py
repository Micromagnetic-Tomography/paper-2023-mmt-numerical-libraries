import mmt_micromagnetic_demag_signature as mds
import numpy as np
nm, µm = 1e-9, 1e-6

# Load the MERRILL files from random state 2 of Cortes et al. (2022):
FILE_energy = './grain_OPX042_rnd2.log'  # To read the magnetization
SAVENAME = 'scan_signal_rnd2_scan-height_1000nm.npy'
VBOXFILE = './mag_vol_rnd2.vbox'  # data of FE mesh and M field
# Scan limits are given by lower-left and upper-right sensor positions
scan_spacing = (10 * nm, 10 * nm)
scan_limits = np.array([[-1.5, -1.5], [1.5, 1.5]]) * µm
scan_height = 1000 * nm
demag_signal = mds.MicroDemagSignature(scan_limits, scan_spacing, scan_height,
                                       VBOXFILE, FILE_energy)
# Shift the geom center of the grain to the origin
demag_signal.read_input_files(origin_to_geom_center=True)
# Compute stray field signal
demag_signal.compute_scan_signal(method='cython')
np.save(SAVENAME, demag_signal.Bz_grid)
