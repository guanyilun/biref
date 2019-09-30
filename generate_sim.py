"""This script generates simulations with birefringence
effects included"""
import os, numpy as np

from pixell import enmap
from actsims.simgen import SimGen

###################
# user parameters #
###################

# number of simulations
nsims = 1

# sim set and idx
sim_id_0 = 100
set_id = 0

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

output_dir = "/global/cscratch1/sd/yguan/sims/v0.5/array"
alpha_map_dir = "/global/cscratch1/sd/yguan/sims/v0.5/alpha"

# use mpi
use_mpi = True
# use_mpi = False

# sim parameter
max_cached  = 0
map_version = 'v6.2.0_calibrated_mask_version_padded_v1'
# apply_rotation = False
apply_rotation = True

################
# main program #
################

if use_mpi:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    indices = np.array_split(np.arange(nsims), size)
    core_start = int(indices[rank][0]) + sim_id_0
    core_end = int(indices[rank][-1]) + 1 + sim_id_0
    print("rank: %d\t %s -> %s" % (rank, core_start, core_end))
else:
    core_start = sim_id_0
    core_end = core_start + nsims
    rank = 0

if rank == 0:
    if not os.path.exists(output_dir):
        print("%s doesn't exist, creating now..." % output_dir)
        os.makedirs(output_dir)
if use_mpi:
    comm.Barrier()

# determine how to generate seed
generate_seed = lambda sim_id: sim_id

for sim_id in np.arange(core_start, core_end):
    print("Looking at set=%d sim_id=%d" % (set_id, sim_id))

    # check if simulation already exists
    postfix = "set%d_id%d.fits" % (set_id, sim_id)
    if apply_rotation:
        filename = "fullskyalpha_%s" % postfix
        filename = os.path.join(alpha_map_dir, filename)
        if not os.path.isfile(filename):
            # generate full sky rotation map
            print("File doesn't exist!")
            raise Exception
        else:
            alpha_map = enmap.read_map(filename)
    else:  # if rotation is not needed, alpha_map will not be useful
        alpha_map = None

    # generate rotated lensed cmb + noise + foregrounds + beam sims at act patches
    sg = SimGen(version=map_version, cmb_type='LensedUnabberatedCMB',
                apply_rotation=apply_rotation, alpha_map=alpha_map, max_cached=max_cached)
    for psa in psa_list:
        print("Generating sim for %s" % psa)
        map_patch = sg.get_sim(**psa, sim_num=sim_id)
        filename = "{patch}_{season}_{array}_{postfix}".format(patch=psa["patch"], season=psa["season"], array=psa["array"], postfix=postfix)
        filename = os.path.join(output_dir, filename)
        print("Saving data: %s..." % filename)
        enmap.write_map(filename, map_patch)
        del map_patch
    del alpha_map
