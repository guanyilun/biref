"""This script generates a hit map based on the hit maps from
different arrays, frequencies, and patches. """

import os
import os.path as op
from pixell import enmap
import numpy as np


# paramter
input_dir = '/project/projectdirs/act/data/synced_maps/mr3f_20190502'
patch = 'deep56'
alpha_error = 0.6 / 180 * np.pi

deep56_files = [
    's14_deep56_pa1_f150_nohwp_night_3pass_4way_coadd_hits.fits',
    's14_deep56_pa2_f150_nohwp_night_3pass_4way_coadd_hits.fits',
    's15_deep56_pa1_f150_nohwp_night_3pass_4way_coadd_hits.fits',
    's15_deep56_pa2_f150_nohwp_night_3pass_4way_coadd_hits.fits',
    's15_deep56_pa3_f090_nohwp_night_3pass_4way_coadd_hits.fits',
    's15_deep56_pa3_f150_nohwp_night_3pass_4way_coadd_hits.fits',
]

boss_files = [
    's15_boss_pa1_f150_nohwp_night_3pass_4way_coadd_hits.fits',
    's15_boss_pa2_f150_nohwp_night_3pass_4way_coadd_hits.fits',
    's15_boss_pa3_f090_nohwp_night_3pass_4way_coadd_hits.fits',
    's15_boss_pa3_f150_nohwp_night_3pass_4way_coadd_hits.fits',
]

# a sample sim just to extract the shape and wcs. This is because
# the data map has a different size as the sims. 
sample_sim_deep56 = enmap.read_map('/global/cscratch1/sd/yguan/sims/v0.3/arrays/deep56_s15_pa1_set0_id0.fits')
sample_sim_boss = enmap.read_map('/global/cscratch1/sd/yguan/sims/v0.3/arrays/boss_s15_pa1_set0_id0.fits')

# get Nhits map
first = True
for f in deep56_files:
    filename = op.join(input_dir, f)
    imap = enmap.extract(enmap.read_map(filename), sample_sim_deep56.shape, sample_sim_deep56.wcs)
    if first:
        omap = imap.copy()
        first = False
    else:
        omap += imap
hits_map = omap

# seed the random number
np.random.seed(10)

# generate alpha with standard gaussian
alpha_0 = np.random.randn(*omap.shape)*alpha_error
alpha_map_0 = enmap.samewcs(alpha_0, omap)
alpha_map = alpha_map_0 * hits_map**-0.5
alpha_map[~np.isfinite(alpha_map)] = 0
enmap.write_map("relrot_deep56.fits", alpha_map)

# get Nhits map
first = True
for f in boss_files:
    filename = op.join(input_dir, f)
    imap = enmap.extract(enmap.read_map(filename), sample_sim_boss.shape, sample_sim_boss.wcs)
    if first:
        omap = imap.copy()
        first = False
    else:
        omap += imap
hits_map = omap

# seed the random number
np.random.seed(20)

# generate alpha with standard gaussian
alpha_0 = np.random.randn(*omap.shape)*alpha_error
alpha_map_0 = enmap.samewcs(alpha_0, omap)
alpha_map = alpha_map_0 * hits_map**-0.5
alpha_map[~np.isfinite(alpha_map)] = 0
enmap.write_map("relrot_boss.fits", alpha_map)
