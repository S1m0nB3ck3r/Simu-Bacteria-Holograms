# -*- coding: utf-8 -*-

from simu_hologram import *
import numpy as np
import cupy as cp
from cupyx import jit
import os

import math
import time
import matplotlib.pyplot as plt
import datetime

import propagation
import traitement_holo
from PIL import Image  
import PIL
import argparse






#######################################################################################################
#########################################           MAIN        #######################################
#######################################################################################################


if __name__ == "__main__":

    #paramètres
    chemin_base = "c:/TRAVAIL/simu_holo"
    if not os.path.exists(chemin_base):
        os.mkdir(chemin_base)

    nb_holo_to_simulate = 10
    number_of_bacteria = 20

    #volume (taille holo & nombre de plans)
    x_size = 1024
    y_size = 1024
    z_size = 200

    #Phi
    transmission_milieu = 1.0
    index_milieu = 1.33
    index_bactérie = 1.35

    #Camera
    pix_size = 5.5e-6
    grossissement = 40
    vox_size_xy = pix_size / grossissement
    vox_size_z = 100e-6 / z_size

    #parametres source illumination
    moyenne = 1.0
    ecart_type = 0.1
    bruit_gaussien = np.abs(np.random.normal(moyenne, ecart_type, [x_size, y_size]))
    
    wavelenght = 660e-9
    lambda_milieu = wavelenght / index_milieu


    #Creation des repertoires d'enregistrement:
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    print("Date et heure actuelles:", formatted_date_time)

    chemin_positions = os.path.join(chemin_base,formatted_date_time, "positions")
    chemin_holograms = os.path.join(chemin_base,formatted_date_time, "holograms")

    os.makedirs(chemin_positions, exist_ok=True)
    os.makedirs(chemin_holograms, exist_ok=True)

    volume_size = [x_size, y_size, z_size]


    shift_in_env = 0.0
    shift_in_obj = 2.0 * cp.pi * vox_size_z * (index_bactérie - index_milieu) / wavelenght
    transmission_in_obj = 0.0

    #allocations
    h_holo = np.zeros(shape = (x_size, y_size), dtype = np.float32)
    d_holo = cp.zeros(shape = (x_size, y_size), dtype = cp.float32)
    d_fft_holo = cp.zeros(shape = (x_size, y_size), dtype = cp.complex64)
    d_fft_holo_propag = cp.zeros(shape = (x_size, y_size), dtype = cp.complex64)
    d_holo_propag = cp.zeros(shape = (x_size, y_size), dtype = cp.float32)
    d_KERNEL = cp.zeros(shape = (x_size, y_size), dtype = cp.complex64)



    for n in range(nb_holo_to_simulate):

        #creation du champs d'illumination
        np_field_plane = np.full(shape=[x_size, y_size], fill_value =  0.0+0.0j, dtype=cp.complex64)
        np_field_plane.real = np.sqrt(bruit_gaussien)
        field_plane = cp.asarray(np_field_plane)

        #initialisation du masque (volume 3D booleen présence ou non bactérie)
        mask_volume = np.full(shape = volume_size, fill_value=False, dtype=np.bool8)

        print("generation bacteries n°", n)

        liste_bacteries = gen_random_bacteria(number_of_bact = number_of_bacteria,
                            xyz_min_max=[0, x_size * vox_size_xy, 0, y_size * vox_size_xy, 0, z_size * vox_size_z],
                            thickness=1e-6, length=3e-6)

        print("insertion bacteries dans volume n°", n)

        for i in range (len(liste_bacteries)):
            liste_bacteries[i].to_file(chemin_positions + "/bact_" + str(n) + ".txt")

        #insertion des bactéries dans le volume
        for i in range (len(liste_bacteries)):
            insert_bact_in_mask_volume(mask_volume, liste_bacteries[i], vox_size_xy, vox_size_z)
            print("bact " + str(i) + " ok")
        
        #invertion de l'axe Z du volume (puisqu'à la localisation la propagation est inversée)
        mask_volume =cp.flip(mask_volume, axis=2)

        print("propagation volume n°", n)

        #SIMU PROPAGATION
        for i in range(mask_volume.shape[2]):
            
            field_plane = propagation.propag_angular_spectrum(field_plane, d_fft_holo, d_KERNEL, d_fft_holo_propag, d_holo_propag,
                                                lambda_milieu, 40.0, pix_size, x_size, y_size, vox_size_z, 0,0)
            
            maskplane = cp.asarray(mask_volume[:,:,i])

            field_plane = phase_shift_through_plane(mask_plane=maskplane, plane_to_shift=field_plane,
                                                    shift_in_env=shift_in_env, shift_in_obj=shift_in_obj)
            
        traitement_holo.save_image(traitement_holo.intensite(field_plane), chemin_holograms + "/holo_" + str(n) + ".bmp")

    # traitement_holo.affichage(traitement_holo.intensite(field_plane))
