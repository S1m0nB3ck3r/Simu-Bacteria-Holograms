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
    chemin_base = os.getcwd() + "/simu_bact_liste"
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
    thickness = 1.0e-6
    length = 3.0e-6

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


    #creation du champs d'illumination
    np_field_plane = np.full(shape=[x_size, y_size], fill_value =  0.0+0.0j, dtype=cp.complex64)
    bruit_gaussien = np.abs(np.random.normal(moyenne, 0.1, [x_size, y_size]))
    np_field_plane.real = np.sqrt(bruit_gaussien)
    field_plane = cp.asarray(np_field_plane)

    #initialisation du masque (volume 3D booleen présence ou non bactérie)
    mask_volume = np.full(shape = volume_size, fill_value=False, dtype=np.bool8)
    
    positions_bact = [
        [100*vox_size_xy, 100*vox_size_xy, 50*vox_size_z, 0.0, 0.0],
        [200*vox_size_xy, 200*vox_size_xy, 60*vox_size_z, 0.0, 10.0],
        [300*vox_size_xy, 300*vox_size_xy, 70*vox_size_z, 0.0, 20.0],
        [400*vox_size_xy, 400*vox_size_xy, 80*vox_size_z, 0.0, 30.0],
        [500*vox_size_xy, 500*vox_size_xy, 90*vox_size_z, 0.0, 40.0],
        [600*vox_size_xy, 600*vox_size_xy, 100*vox_size_z, 0.0, 50.0],
        [700*vox_size_xy, 700*vox_size_xy, 110*vox_size_z, 0.0, 60.0],
        [800*vox_size_xy, 800*vox_size_xy, 120*vox_size_z, 0.0, 70.0],
        [900*vox_size_xy, 900*vox_size_xy, 130*vox_size_z, 0.0, 80.0],
        [1000*vox_size_xy, 1000*vox_size_xy, 140*vox_size_z, 0.0, 90.0]
    ]

    #creation des bactéries
    bacteries = []
    with open(chemin_positions + "/bact_liste.txt", "a") as file:
        for b in positions_bact:
            #instantiation bacterie
            bact = Bacterie(
                pos_x=b[0] + border*vox_size_xy, pos_y=b[1] + border*vox_size_xy, pos_z=b[2], thickness=thickness, length=length, theta=b[3], phi=b[4]
            )
            #ecriture fichier txt
            txt = "{posx}\t{posy}\t{posz}\t{lengh}\t{thickness}\t{angle1}\t{angle2}\t\n".format(
                    posx = b[0],
                    posy = b[1],
                    posz = b[2],
                    lengh = length, thickness = thickness, angle1 = b[3], angle2 = b[3]
                )
            file.write(txt)
            bacteries.append(bact)

    print("insertion bacteries dans volume")
        #insertion des bactéries dans le volume
    for i in range (len(bacteries)):
        print("bact " + str(i) + " ok")
        insert_bact_in_mask_volume(mask_volume, bacteries[i], vox_size_xy, vox_size_z, 1)
        
    #invertion de l'axe Z du volume (puisqu'à la localisation la propagation est inversée)
    mask_volume =cp.flip(mask_volume, axis=2)

    print("propagation volume")

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
            
        traitement_holo.save_image(traitement_holo.intensite(field_plane)[border:border+holo_size_xy, border:border+holo_size_xy], chemin_holograms + "/holo_simu_bact_.bmp")
    print("positions :")
    for b in positions_bact:
        print(b)
    # traitement_holo.affichage(traitement_holo.intensite(field_plane))
