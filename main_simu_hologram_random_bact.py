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
    
    #nombre d'hologrammes à simuler
    nb_holo_to_simulate = 100
    number_of_bacteria = 200

    #volume (taille holo & nombre de plans)
    #on prend la taille x et y 2x plus grand, pour cropper l'hologramme final afin de ne pas avoir l'effet "périodique" dû aux transformées de fourier
    border = 256
    holo_size_xy = 1024
    holo_size_xy_w_b = holo_size_xy + border * 2
    upscale_factor = 2
    holo_size_xy_upscaled = holo_size_xy * upscale_factor
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

    parameters = {
    'holo_size_x': holo_size_xy,
    'holo_size_y': holo_size_xy,
    'holo_plane_number': z_size, 
    'medium_index': index_milieu, 
    'object_index': index_bactérie, 
    'pix_size_cam': pix_size, 
    'magnification_cam': grossissement, 
    'Z_step': vox_size_z, 
    'illumination_wavelength': wavelenght, 
    'medium_wavelength': lambda_milieu
    }


    #Creation des repertoires d'enregistrement:
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    print("Date et heure actuelles:", formatted_date_time)

    chemin_positions = os.path.join(chemin_base,formatted_date_time, "positions")
    chemin_holograms = os.path.join(chemin_base,formatted_date_time, "holograms")
    chemin_data_holo = os.path.join(chemin_base,formatted_date_time, "data_holograms")

    os.makedirs(chemin_positions, exist_ok=True)
    os.makedirs(chemin_holograms, exist_ok=True)
    os.makedirs(chemin_data_holo, exist_ok=True)

    volume_size = [holo_size_xy, holo_size_xy, z_size]
    volume_size_w_b = [holo_size_xy_w_b, holo_size_xy_w_b, z_size]
    volume_size_upscaled = [holo_size_xy * upscale_factor, holo_size_xy * upscale_factor, z_size]   

    shift_in_env = 0.0
    shift_in_obj = 2.0 * cp.pi * vox_size_z * (index_bactérie - index_milieu) / wavelenght
    transmission_in_obj = 0.0

    #allocations
    h_holo = np.zeros(shape = (holo_size_xy_w_b, holo_size_xy_w_b), dtype = np.float32)
    d_holo = cp.zeros(shape = (holo_size_xy_w_b, holo_size_xy_w_b), dtype = cp.float32)
    d_fft_holo = cp.zeros(shape = (holo_size_xy_w_b, holo_size_xy_w_b), dtype = cp.complex64)
    d_fft_holo_propag = cp.zeros(shape = (holo_size_xy_w_b, holo_size_xy_w_b), dtype = cp.complex64)
    d_holo_propag = cp.zeros(shape = (holo_size_xy_w_b, holo_size_xy_w_b), dtype = cp.float32)
    d_KERNEL = cp.zeros(shape = (holo_size_xy_w_b, holo_size_xy_w_b), dtype = cp.complex64)

    rnd = np.random.default_rng()

    for n in range(nb_holo_to_simulate):
        print("propagation plan n°", n+1)

        data_file = os.path.join(chemin_data_holo, "data_" + str(n) + ".npz")
        holo_file = os.path.join(chemin_holograms, "holo_" + str(n) + ".bmp")
        positions_file = os.path.join(chemin_positions, "bact_" + str(n) + ".txt")

        #creation du champs d'illumination
        np_field_plane = np.full(shape=[holo_size_xy_w_b, holo_size_xy_w_b], fill_value =  0.0+0.0j, dtype=cp.complex64)
        ecart_type_bruit = (ecart_type_min_max[max] - ecart_type_min_max[min]) * rnd.random() + ecart_type_min_max[min]
        bruit_gaussien = np.abs(np.random.normal(moyenne, ecart_type_bruit, [holo_size_xy_w_b, holo_size_xy_w_b]))
        np_field_plane.real = np.sqrt(bruit_gaussien)
        cp_field_plane = cp.asarray(np_field_plane)

        #initialisation du masque (volume 3D booleen présence ou non bactérie)
        cp_mask_volume = cp.full(shape = volume_size, fill_value=0, dtype=cp.float16)
        cp_mask_plane_w_border = cp.full(shape = (holo_size_xy_w_b, holo_size_xy_w_b ), fill_value=0.0, dtype=cp.float32)
        cp_mask_volume_upscaled = cp.full(shape = volume_size_upscaled, fill_value=0, dtype=cp.float16)

        # np_mask_volume_upscaled = cp.full(shape = volume_size_upscaled, fill_value=0, dtype=cp.float16)
        # np_mask_plane_w_border = cp.full(shape = (holo_size_xy_w_b, holo_size_xy_w_b ), fill_value=0.0, dtype=cp.float32)

        #génération des bactéries
        print("generation bacteries n°", n)
        liste_bacteries = gen_random_bacteria(
            number_of_bact = number_of_bacteria,
            xyz_min_max = [ 0, holo_size_xy * vox_size_xy, 0, holo_size_xy * vox_size_xy, 0, z_size * vox_size_z ],
            thickness_min_max = epaisseur_min_max,
            length_min_max = longueur_min_max
            )

        print("insertion bacteries dans volume n°", n)
        for bact in liste_bacteries:
            bact.to_file(positions_file)
        
        bacteria_list = [
            {
                "thickness": b.thickness,
                "length": b.length,
                "x_position_m": b.pos_x,
                "y_position_m": b.pos_y,
                "z_position_m": b.pos_z,
                "theta_angle": b.theta,
                "phi_angle": b.phi
                }
                for b in liste_bacteries
                ]

        #insertion des bactéries dans le volume
        for i in range (len(liste_bacteries)):
            print("bact " + str(i) + " ok")
            GPU_insert_bact_in_mask_volume(cp_mask_volume_upscaled, liste_bacteries[i], vox_size_xy / upscale_factor, vox_size_z)
            # insert_bact_in_mask_volume(np_mask_volume_upscaled, liste_bacteries[i], vox_size_xy / upscale_factor, vox_size_z)
        
        # cp_mask_volume_upscaled = cp.asarray(np_mask_volume_upscaled)
        #invertion de l'axe Z du volume (puisqu'à la localisation la propagation est inversée)
        cp_mask_volume_upscaled = cp.flip(cp_mask_volume_upscaled, axis=2)

        cp_mask_volume = cp_mask_volume_upscaled[:,:,:].reshape(holo_size_xy, upscale_factor, holo_size_xy, upscale_factor, z_size).mean(axis=(1,3))

        print("propagation")
        #SIMU PROPAGATION
        for i in range(z_size):
           
            
            cp_field_plane = propagation.propag_angular_spectrum(cp_field_plane, d_fft_holo, d_KERNEL, d_fft_holo_propag, d_holo_propag,
                                                lambda_milieu, 40.0, pix_size, holo_size_xy_w_b, holo_size_xy_w_b, vox_size_z, 0,0)           

            cp_mask_plane_w_border = pad_centered(cp_mask_volume[:,:,i], [holo_size_xy_w_b, holo_size_xy_w_b])

            cp_field_plane = phase_shift_through_plane(mask_plane=cp_mask_plane_w_border, plane_to_shift=cp_field_plane,
                                                    shift_in_env=shift_in_env, shift_in_obj=shift_in_obj)
            

        croped_field_plane = cp_field_plane[border:border+holo_size_xy, border:border+holo_size_xy]


        save_holo_data(
            data_file,
            cp.asnumpy(cp_mask_volume != 0.0),  cp.asnumpy(traitement_holo.intensite(croped_field_plane)),
            parameters, bacteria_list
            )
        
        # plan = cp.asnumpy((cp_mask_volume != 0.0)).sum(axis=2)

        # plt.imshow(plan, cmap="gray")
        # plt.title("Projection Z (masque)")
        # plt.colorbar()
        # plt.show()
        
        # bool_vol, t2, t3, t4 = load_holo_data(data_file)

        # print("non zero :", np.count_nonzero(bool_vol.astype(np.float32) != 0.0))

        # plan = cp.asnumpy(bool_vol).sum(axis=2)

        # plt.imshow(plan, cmap="gray")
        # plt.title("Projection Z (masque)")
        # plt.colorbar()
        # plt.show()