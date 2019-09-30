"""This script generates simulations with birefringence
effects included"""

import os, numpy as np

from pixell import utils, curvedsky, enmap

###################
# user parameters #
###################

# number of simulations
nsims = 102

# sim set and idx
sim_id_0 = 0
set_id = 0

# rotation map configuration
pixel_size = 0.5  # arcmin
lmax = 8000
A_CB = 0.3

alpha_map_dir = "/global/cscratch1/sd/yguan/sims/v0.4/alpha"

# force regeneration of alpha maps
force = True

# use mpi
use_mpi = True

################
# main program #
################

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
    rank = 0

if rank == 0:
    if not os.path.exists(alpha_map_dir):
        print("%s doesn't exist, creating now..." % alpha_map_dir)
        os.makedirs(alpha_map_dir)

if use_mpi:
    comm.Barrier()

# determine how to generate seed
generate_seed = lambda sim_id: sim_id

# generate power spectrum for alpha
ell = np.arange(0, lmax+1)
ps_alpha = A_CB*1E-4*2*np.pi/(ell*(ell+1))
ps_alpha[0] = 0

shape, wcs = enmap.fullsky_geometry(pixel_size * utils.arcmin)

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
    else: pass
