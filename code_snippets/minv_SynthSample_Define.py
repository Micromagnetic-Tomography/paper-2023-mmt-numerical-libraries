import numpy as np
from pathlib import Path
import json

# -----------------------------------------------------------------------------
# DEFINE SAMPLE: Save particle and scan surface properties

# Directory to save the files
SaveDir = Path('SyntheticSampleFiles')
SaveDir.mkdir(exist_ok=True)

particles = np.loadtxt('area1_ums_part_centers_vols.txt')
# Scale the pos and vols by mu-m before saving in npz file
np.savez(SaveDir / 'Area1_UMS_NPZ_ARRAYS',
         particle_positions=particles[:, :3] * 1e-6,
         volumes=particles[:, 3] * 1e-18)

# Set dictionary with scanning surface parameters
metadict = {}
metadict["Scan height Hz"] = 2e-6
metadict["Scan area x-dimension Sx"] = 351 * 1e-6
metadict["Scan area y-dimension Sy"] = 201 * 1e-6
metadict["Scan x-step Sdx"] = 1e-6
metadict["Scan y-step Sdy"] = 1e-6
metadict["Time stamp"] = '0000'
metadict["Number of particles"] = 8
# Important! for area sensors:
metadict["Sensor dimensions"] = (0.5e-6, 0.5e-6)

# Save dictionary into json file
with open(SaveDir / "AREA1_UMS_METADICT.json", 'w') as f:
    json.dump(metadict, f)
