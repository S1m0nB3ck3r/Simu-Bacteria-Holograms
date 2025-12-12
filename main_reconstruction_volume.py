# -*- coding: utf-8 -*-
from traitement_holo import *
from propagation import *


import matplotlib.pyplot as plt
import os
from PIL import Image  
import PIL
import numpy as np
import cupy as cp


#Pramètres
transmission_milieu = 1.0
index_milieu = 1.33
pix_size = 5.5e-6
magnification = 40
vox_size_xy = pix_size / magnification
vox_size_z = 100e-6
wavelenght = 660e-9
lambda_milieu = wavelenght / index_milieu

#paramètres reconstruction
holo_size_xy = 1024
distance_propag_ini = 0e-6
dz = 1e-6
np_plan_propag = 100
volume_size = holo_size_xy * np_plan_propag

#allocations
h_HOLO = np.zeros(shape = (holo_size_xy, holo_size_xy), dtype = np.float32)
d_HOLO = cp.zeros(shape = (holo_size_xy, holo_size_xy), dtype = cp.float32)
d_FFT_HOLO = cp.zeros(shape = (holo_size_xy, holo_size_xy), dtype = cp.complex64)
d_FFT_HOLO_PROPAG = cp.zeros(shape = (holo_size_xy, holo_size_xy), dtype = cp.complex64)
d_holo_propag = cp.zeros(shape = (holo_size_xy, holo_size_xy), dtype = cp.float32)
d_KERNEL = cp.zeros(shape = (holo_size_xy, holo_size_xy), dtype = cp.complex64)
d_HOLO_VOLUME_PROPAG_MODULE = cp.full(shape = (holo_size_xy, holo_size_xy, np_plan_propag), fill_value=False, dtype=np.float32)
d_HOLO_VOLUME_PROPAG_CPLX = cp.full(shape = (holo_size_xy, holo_size_xy, np_plan_propag), fill_value=False, dtype=np.complex64)


#start reconstruction
holo_path = "./holo_simu.bmp"

h_HOLO = read_image(holo_path)
d_HOLO = cp.array(h_HOLO)

output = "CPLX" # "MODULE" or "CPLX"
if output == "MODULE":
    volume_propag_angular_spectrum_to_module(d_HOLO, d_FFT_HOLO, d_KERNEL, d_FFT_HOLO_PROPAG, d_HOLO_VOLUME_PROPAG_MODULE,
                                             lambda_milieu, magnification, pix_size, holo_size_xy, holo_size_xy, distance_propag_ini, dz, np_plan_propag, 0.0, 0.0)
    
    #affichage plan XY à Z = 50
    display(d_HOLO_VOLUME_PROPAG_MODULE[:,:,50])

    #affichage plan XZ à Y = 512
    display(d_HOLO_VOLUME_PROPAG_MODULE[:,512,:])

else:
    volume_propag_angular_spectrum_complex(d_HOLO, d_FFT_HOLO, d_KERNEL, d_FFT_HOLO_PROPAG, d_HOLO_VOLUME_PROPAG_CPLX, 
                                           lambda_milieu, magnification, pix_size, holo_size_xy, holo_size_xy, distance_propag_ini, dz, np_plan_propag, 0.0, 0.0)
    
    #affichage plan XY à Z = 50
    display(intensite(d_HOLO_VOLUME_PROPAG_CPLX[:,:,50]))

    #affichage plan XZ à Y = 512
    display(intensite(d_HOLO_VOLUME_PROPAG_CPLX[:,512,:]))


