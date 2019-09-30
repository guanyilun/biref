"""This script aims to generate rotated simulations by rotating
the k-space combined IQU maps by a fixed angle"""
import numpy as np
from pixell import enmap

# parameters
input_dir = "/global/cscratch1/sd/yguan/sims/v0.5/iqu_unrot"
output_dir = "/global/cscratch1/sd/yguan/sims/v0.5/iqu_relrot"
# psa = "s14&15_boss"
psa = "s14&15_deep56"
nsims = 101

# absolute rotation angle
# alpha = enmap.read_map("/global/cscratch1/sd/yguan/sims/v0.3/relrot_boss.fits")
alpha = enmap.read_map("/global/cscratch1/sd/yguan/sims/v0.3/relrot_deep56.fits")

# utility functions
def load_map(sim_id, map_type, psa):
    """A utility function to load map of a given type"""
    prefix = "dataCoadd_combinedSimset00"
    filename = f"{input_dir}/{prefix}_{sim_id:05d}_{map_type}_{psa}.fits"
    print("Loading:", filename)
    return enmap.read_map(filename)

def write_map(omap, sim_id, map_type, psa):
    """A utility function to load map of a given type"""
    prefix = "dataCoadd_combinedSimset00"
    filename = f"{output_dir}/{prefix}_{sim_id:05d}_{map_type}_{psa}.fits"
    print("Saving:", filename)
    return enmap.write_map(filename, omap)

# main function

c, s = np.cos(2*alpha), np.sin(2*alpha)
for sim_id in range(nsims):
    # load the IQU maps for each sim
    qmap = load_map(sim_id, "Q", psa)
    umap = load_map(sim_id, "U", psa)

    # rotate QU maps by a fixed angle
    qmap_rot = qmap.copy()
    umap_rot = umap.copy()

    qmap_rot = c*qmap-s*umap
    umap_rot = s*qmap+c*umap

    # save the rotated QU maps
    write_map(qmap_rot, sim_id, "Q", psa)
    write_map(umap_rot, sim_id, "U", psa)

# copy the I maps over
import os
os.system(f"cp -v {input_dir}/*_I_*.fits {output_dir}/")

print("Done!")
