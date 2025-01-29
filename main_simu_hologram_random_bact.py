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
    chemin_base = os.getcwd() + "/simu_bact_random"
    if not os.path.exists(chemin_base):
        os.mkdir(chemin_base)

    nb_holo_to_simulate = 1000
    number_of_bacteria = 50

    #volume (taille holo & nombre de plans)
    #on prend la taille x et y 2x plus grand, pour cropper l'hologramme final afin de ne pas avoir l'effet "périodique" dû aux transformées de fourier
    border = 512
    holo_size_xy = 1024
    x_size = holo_size_xy + border * 2
    y_size = holo_size_xy + border * 2
    z_size = 200

    #Pramètres bactéries
    transmission_milieu = 1.0
    index_milieu = 1.33
    index_bactérie = 1.335
    longueur_min_max = {min : 3.0e-6, max: 4.0e-6}
    epaisseur_min_max = {min : 1.0e-6, max: 2.0e-6}

    #Camera
    pix_size = 5.5e-6
    grossissement = 40
    vox_size_xy = pix_size / grossissement
    vox_size_z = 100e-6 / z_size

    #parametres source illumination
    moyenne = 1.0
    ecart_type_min_max = {min : 0.01, max : 0.1}
    
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

    rnd = np.random.default_rng()

    for n in range(nb_holo_to_simulate):

        #creation du champs d'illumination
        np_field_plane = np.full(shape=[x_size, y_size], fill_value =  0.0+0.0j, dtype=cp.complex64)
        ecart_type_bruit = (ecart_type_min_max[max] - ecart_type_min_max[min]) * rnd.random() + ecart_type_min_max[min]
        bruit_gaussien = np.abs(np.random.normal(moyenne, ecart_type_bruit, [x_size, y_size]))
        np_field_plane.real = np.sqrt(bruit_gaussien)
        field_plane = cp.asarray(np_field_plane)

        #initialisation du masque (volume 3D booleen présence ou non bactérie)
        mask_volume = np.full(shape = volume_size, fill_value=False, dtype=np.bool8)

        #génération des bactéries
        print("generation bacteries n°", n)
        liste_bacteries = gen_random_bacteria(
            number_of_bact = number_of_bacteria,
                            xyz_min_max=[
                                border * vox_size_xy, (x_size - border) * vox_size_xy,
                                border * vox_size_xy, (y_size - border) * vox_size_xy,
                                  0, z_size * vox_size_z
                                  ],
            thickness_min_max = epaisseur_min_max,
            length_min_max = longueur_min_max
            )

        print("insertion bacteries dans volume n°", n)
        with open(chemin_positions + "/bact_" + str(n) + ".txt", "a") as file:
            for bact in liste_bacteries:

                txt = "{posx}\t{posy}\t{posz}\t{lengh}\t{thickness}\t{angle1}\t{angle2}\t\n".format(
                    posx = bact.pos_x - border*vox_size_xy,
                    posy = bact.pos_y - border*vox_size_xy,
                    posz = bact.pos_z,
                    lengh = bact.length, thickness = bact.thickness, angle1 = bact.theta, angle2 = bact.phi
                )
                file.write(txt)

        #insertion des bactéries dans le volume
        for i in range (len(liste_bacteries)):
            print("bact " + str(i) + " ok")
            insert_bact_in_mask_volume(mask_volume, liste_bacteries[i], vox_size_xy, vox_size_z, 1)
        
        #invertion de l'axe Z du volume (puisqu'à la localisation la propagation est inversée)
        mask_volume =cp.flip(mask_volume, axis=2)

        print("propagation volume n°", n)

        #SIMU PROPAGATION
        for i in range(mask_volume.shape[2]):
            
            field_plane = propagation.propagate_angular_spectrum(
                input_wavefront=field_plane, kernel=d_KERNEL,
                wavelength=lambda_milieu, magnification=40.0,
                pixel_size=pix_size, width=x_size, height=y_size,
                propagation_distance=vox_size_z, min_frequency=0,max_frequency=0)
            
            maskplane = cp.asarray(mask_volume[:,:,i])

            field_plane = phase_shift_through_plane(mask_plane=maskplane, plane_to_shift=field_plane,
                                                    shift_in_env=shift_in_env, shift_in_obj=shift_in_obj)
            
        traitement_holo.save_image(traitement_holo.intensite(field_plane)[border:border+holo_size_xy, border:border+holo_size_xy], chemin_holograms + "/holo_simu_bact_" + str(n) + ".bmp")

    # traitement_holo.affichage(traitement_holo.intensite(field_plane))
