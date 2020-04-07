from __future__ import print_function

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    has_mpi = True
except ImportError:
    rank = 0
    size = 1
    has_mpi = False
import time
from two_color import utils
from IPython import embed
from cxid9114.sim import sim_utils

from cxid9114.sf import struct_fact_special
import os
import numpy as np
from scipy import constants
from copy import deepcopy

from cctbx import sgtbx, miller
from cctbx.crystal import symmetry
from dials.array_family import flex
from dials.command_line.find_spots import phil_scope as strong_phil_scope
from dials.algorithms.indexing.compare_orientation_matrices import rotation_matrix_differences
import dxtbx
from dxtbx.model.experiment_list import ExperimentList, Experiment, ExperimentListFactory
from dxtbx.model.beam import BeamFactory
from dxtbx.model.crystal import CrystalFactory
from dxtbx.model.detector import DetectorFactory
from scitbx.matrix import sqr, col
from simtbx.nanoBragg import nanoBragg
from simtbx.nanoBragg import shapetype

from cxid9114.helpers import compare_with_ground_truth

from two_color.utils import get_two_color_rois
from two_color.two_color_grid_search import two_color_grid_search
from two_color.two_color_phil import params as index_params
from cxid9114.utils import open_flex

num_imgs = 1

import os
datadir = os.environ.get("DD3")
#DD = "/net/dials/raid1/dermen/cyto/"


######################
# LOAD JUNGFRAU MODEL
######################

print("Load the big azz detector")
expList_for_DET = ExperimentListFactory.from_json_file(
    os.path.join(datadir, "refined_final.expt"),
    check_format=False)
DETECTOR = expList_for_DET.detectors()[0]
image_shape = (len(DETECTOR),) + DETECTOR[0].get_image_size()

###
# load the beam and the crystal from the refined experiment
print("Opening refined exp")
refined_El = ExperimentListFactory.from_json_file("/net/dials/raid1/dermen/cyto/idx-run_000795.JF07T32V01_master_00344_integrated.expt")
BEAM = refined_El.beams()[0]
iset = refined_El.imagesets()[0]
master_file_path = iset.get_path(0)
print("Opening master load")
master_loader = dxtbx.load(master_file_path)
master_file_idx = iset.indices()[0]
SPECTRUM_BEAM = master_loader.get_beam(master_file_idx)
print(SPECTRUM_BEAM)
wavelen = SPECTRUM_BEAM.get_wavelength()

en_slice = slice(500, 2250, 40)
en_slice = slice(1500, 1600, 40)

from cxid9114.parameters import ENERGY_CONV
energies = SPECTRUM_BEAM.get_spectrum_energies().as_numpy_array()
energies = energies[en_slice]
wavelens = ENERGY_CONV/energies
fluxes = SPECTRUM_BEAM.get_spectrum_weights().as_numpy_array()
fluxes =fluxes[en_slice]
fluxes /= fluxes.sum()
fluxes *= 1e12
SPECTRUM_BEAM.set_flux(1e12)

#############
# BEAM MODEL
#############
#beam_descr={'direction': (-0.0, -0.0, 1.0),
# 'wavelength': 1.3037,
# 'divergence': 0.0,
# 'sigma_divergence': 0.0,
# 'polarization_normal': (0.0, 1.0, 0.0),
# 'polarization_fraction': 0.999,
# 'flux': 1e12,
# 'transmission': 1.0}
#
## two color experiment, two energies 100 eV apart
#ENERGYLOW = 9510
#ENERGYHIGH = 9610
#ENERGY_CONV = 1e10*constants.c*constants.h / constants.electron_volt
#WAVELENLOW = ENERGY_CONV/ENERGYLOW
#WAVELENHIGH = ENERGY_CONV/ENERGYHIGH
#BEAM = BeamFactory.from_dict(beam_descr)

from iotbx.reflection_file_reader import any_reflection_file
struc_fac_cif = os.path.join(datadir, "5wp2-sf.cif")
Famp = any_reflection_file(struc_fac_cif).as_miller_arrays()[0]
assert(Famp.is_xray_amplitude_array())

#a = Famp.unit_cell().parameters()[0]  # 77
#c = Famp.unit_cell().parameters()[2]  # 263
CRYSTAL = refined_El.crystals()[0]

# make a single pattern
if rank == 0:
    print("Begin the big azz simulation")

outname = "%s/cytojung_h5_rank%d.h5" % (datadir, rank)

# pick out how many images per file ...
imgs_per = {}
for r in range(size):
    imgs_per[r] = 0
for i in range(num_imgs):
    if i % size != rank:
        continue
    imgs_per[rank] += 1

# make them images, allocate a hdf5 file per rank
with utils.H5AttributeGeomWriter(outname, image_shape=image_shape,
                                 num_images=imgs_per[rank], detector=DETECTOR, beam=BEAM) as writer:

    for img_num in range(num_imgs):
        if img_num % size != rank:
            continue
        ###############################
        # MAKE THE RANDOM CRYSTAL ORI
        ###############################

        #np.random.seed(3142019 + img_num)
        ## make random rotation about principle axes
        #x = col((-1, 0, 0))
        #y = col((0, -1, 0))
        #z = col((0, 0, -1))
        #rx, ry, rz = np.random.uniform(-180, 180, 3)
        #RX = x.axis_and_angle_as_r3_rotation_matrix(rx, deg=True)
        #RY = y.axis_and_angle_as_r3_rotation_matrix(ry, deg=True)
        #RZ = z.axis_and_angle_as_r3_rotation_matrix(rz, deg=True)
        #M = RX*RY*RZ
        #real_a = M*col((a, -.5*a, 0))
        #real_b = M*col((0, np.sqrt(3)*.5*a, 0))
        #real_c = M*col((0, 0, c))

        ## dxtbx crystal description
        #cryst_descr = {'__id__': 'crystal',
        #              'real_space_a': real_a.elems,
        #              'real_space_b': real_b.elems,
        #              'real_space_c': real_c.elems,
        #              'space_group_hall_symbol': ' P 4nw 2abw'}

        #CRYSTAL = CrystalFactory.from_dict(cryst_descr)
        ###
        ###

        t = time.time()

        sims = sim_utils.sim_colors(
            CRYSTAL, DETECTOR, SPECTRUM_BEAM, [Famp] + [None]*(len(energies)-1),
            energies,
            fluxes, pids=None, profile="gauss", cuda=False, oversample=0,
            Ncells_abc=(20, 20, 20), mos_dom=1, mos_spread=0,
            exposure_s=1, beamsize_mm=0.001, device_Id=0,
            amorphous_sample_thick_mm=0.200, add_water=True,
            show_params=False, accumulate=False, crystal_size_mm=0.01, printout_pix=None,
            time_panels=True)

        tsim = time.time()-t

        panel_images = np.array(sims[0])

        for pidx in range(len(DETECTOR)):
            SIM = nanoBragg(detector=DETECTOR, beam=SPECTRUM_BEAM, panel_id=pidx)
            SIM.beamsize_mm = 0.001
            SIM.exposure_s = 1
            SIM.flux = 1e12
            SIM.adc_offset_adu = 0
            # SIM.detector_psf_kernel_radius_pixels = 5
            # SIM.detector_psf_type = shapetype.Unknown  # for CSPAD
            SIM.detector_psf_fwhm_mm = 0
            SIM.quantum_gain = 1
            SIM.raw_pixels = flex.double(panel_images[pidx].ravel())
            SIM.add_noise()
            panel_images[pidx] = SIM.raw_pixels.as_numpy_array().reshape(panel_images[pidx].shape)
            SIM.free_all()
            del SIM

        writer.add_image(panel_images)

        if rank == 0:
            print("Done with shot %d / %d , time: %.4fs" % (img_num+1,num_imgs, tsim), flush=True)
