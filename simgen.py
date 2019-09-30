"""This script generates simulations with birefringence
effects included"""

import os, numpy as np

from pixell import utils, curvedsky, enmap
from actsims.simgen import SimGen

###################
# user parameters #
###################

# number of simulations
nsims = 101

# sim set and idx
sim_id_0 = 0
set_id = 0

# rotation map configuration
pixel_size = 0.5  # arcmin
lmax = 8000
A_CB = 1.

# psa of interests
psa_list = [
    {"season": "s14", "patch": "deep56", "array": "pa1"},
    {"season": "s14", "patch": "deep56", "array": "pa2"},
    {"season": "s15", "patch": "deep56", "array": "pa1"},
    {"season": "s15", "patch": "deep56", "array": "pa2"},
    {"season": "s15", "patch": "deep56", "array": "pa3"},
    {"season": "s15", "patch": "boss", "array": "pa1"},
    {"season": "s15", "patch": "boss", "array": "pa2"},
    {"season": "s15", "patch": "boss", "array": "pa3"},
]

output_dir = "/global/cscratch1/sd/yguan/sims/v0.1"
alpha_map_dir = "/global/cscratch1/sd/yguan/sims/v0.1"

# force regeneration of alpha maps
force = False

# use mpi
use_mpi = True

################
# main program #
################

if not os.path.exists(output_dir):
    print("%s doesn't exist, creating now..." % output_dir)
    os.makedirs(output_dir)

# determine how to generate seed
generate_seed = lambda sim_id: sim_id

# generate power spectrum for alpha
ell = np.arange(0, lmax+1)
ps_alpha = A_CB*1E-4*2*np.pi/(ell*(ell+1))
ps_alpha[:2] = 0

shape, wcs = enmap.fullsky_geometry(pixel_size * utils.arcmin)

if use_mpi:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print("Node @ rank=%d\t size=%d" % (rank, size))

    sims_per_core = int(np.ceil(nsims/size))
    core_start = sim_id_0 + rank * sims_per_core
    core_end = core_start + sims_per_core
else:
    core_start = sim_id_0
    core_end = core_start + nsims

for sim_id in np.arange(core_start, core_end):
    print("Looking at set=%d sim_id=%d" % (set_id, sim_id))

    # check if simulation already exists
    postfix = "set%d_id%d.fits" % (set_id, sim_id)
    filename = "fullskyalpha_%s" % postfix
    filename = os.path.join(alpha_map_dir, filename)
    if force or not os.path.isfile(filename):
        # generate full sky rotation map
        seed = generate_seed(sim_id)
        print("Generating random alpha map...")
        alpha_map = curvedsky.rand_map((1,)+shape, wcs, ps_alpha, lmax=lmax, seed=seed)
        # save alpha map
        print("Saving to disk...")
        print("Saving data: %s..." % filename)
        enmap.write_map(filename, alpha_map)
    else:
        print("Found existing alpha map, skip generating...")
        alpha_map = enmap.read_map(filename)

    # generate rotated lensed cmb + noise + foregrounds + beam sims at act patches
    sg = SimGen("v5.3.0_mask_version_padded_v1", cmb_type='LensedUnabberatedCMB', apply_rotation=True, alpha_map=alpha_map)
    for psa in psa_list:
        print("Generating sim for %s" % psa)
        map_patch = sg.get_sim(**psa, sim_num=sim_id)
        filename = "{patch}_{season}_{array}_{postfix}".format(patch=psa["patch"], season=psa["season"], array=psa["array"], postfix=postfix)
        filename = os.path.join(output_dir, filename)
        print("Saving data: %s..." % filename)
        enmap.write_map(filename, map_patch)
        del map_patch
    del alpha_map
