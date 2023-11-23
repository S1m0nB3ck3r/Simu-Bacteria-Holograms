# -*- coding: utf-8 -*-

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

class Bacterie():

    def __init__(self):
        
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0
        self.thickness = 0.0
        self.length = 0.0
        self.theta = 0.0
        self.transmission = 0.0
        self.ref_index = 0.0

    def __init__(self, pos_x, pos_y, pos_z,
                 thickness, length,
                 theta, phi,
                 transmission, ref_index):
        
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.thickness = thickness
        self.length = length
        self.theta = theta
        self.phi = phi
        self.transmission = transmission
        self.ref_index = ref_index

    def to_file(self, path_file):

        txt = "{posx}\t{posy}\t{posz}\t{angle1}\t{angle2}\t\n".format(posx = self.pos_x, posy = self.pos_y, posz = self.pos_z, angle1 = self.theta, angle2 = self.phi)

        with open(path_file, "a") as file:
            file.write(txt)




def gen_random_bacteria(number_of_bact: int, xyz_min_max: list, thickness: float, length: float, transmission: float, ref_index: float):

        # self.pos_x = 0.0
        # self.pos_y = 0.0
        # self.pos_z = 0.0
        # self.thickness = 0.0
        # self.length = 0.0
        # self.theta = 0.0
        # self.transmission = 0.0
        # self.ref_index = 0.0

        rng = np.random.default_rng()

        x_min, x_max, y_min, y_max, z_min, z_max = xyz_min_max

        list_bact = []

        x_positions = (x_max - x_min) * rng.random(number_of_bact) + x_min
        y_positions = (y_max - y_min) * rng.random(number_of_bact) + y_min
        z_positions = (z_max - z_min) * rng.random(number_of_bact) + z_min

        theta_angles = 90.0 * rng.random(number_of_bact)
        phi_angles = 90.0 * rng.random(number_of_bact)
        
        for i in range(number_of_bact):

            list_bact.append(Bacterie(x_positions[i], y_positions[i], z_positions[i], thickness, length,
                                theta_angles[i], phi_angles[i], transmission, ref_index))
        
        return list_bact



def phase_shift_through_plane(mask_plane :cp, plane_to_shift: cp, shift_in_env: float, shift_in_obj: float):

    shift_plane = cp.full(fill_value=shift_in_env, dtype=cp.float32, shape=mask_plane.shape)

    cp.putmask(a=shift_plane, mask=mask_plane, values=shift_in_obj)

    def phase_shift(cplx_plane, shift_plane):

        phase = cp.angle(cplx_plane)
        module = cp.sqrt(cp.real(cplx_plane) ** 2 + cp.imag(cplx_plane) ** 2)

        phase = phase + shift_plane

        return module * cp.exp((0+1.j) * phase)
    
    return phase_shift(plane_to_shift, shift_plane)

def insert_bact_in_mask_volume(mask_volume: np, bact: Bacterie, vox_size_xy: float, vox_size_z: float):
    
    phi_rad = math.radians(bact.phi)
    theta_rad = math.radians(bact.theta)

    #distance Extremité-centre:
    long_Demi_Seg = (bact.length - bact.thickness/2.0) / 2.0

	#calcul des positions des extremités du segment interieur de la bactérie m1 et m2
    m1_x = bact.pos_x - long_Demi_Seg * math.sin(phi_rad) * math.cos(theta_rad)
    m1_y = bact.pos_y - long_Demi_Seg * math.sin(phi_rad) * math.sin(theta_rad)
    m1_z = bact.pos_z - long_Demi_Seg * math.cos(phi_rad)

    m2_x = bact.pos_x + long_Demi_Seg * math.sin(phi_rad) * math.cos(theta_rad)
    m2_y = bact.pos_y + long_Demi_Seg * math.sin(phi_rad) * math.sin(theta_rad)
    m2_z = bact.pos_z + long_Demi_Seg * math.cos(phi_rad)

    #calcul segment [m2 m1]
    m2m1 = np.array([m2_x-m1_x, m2_y-m1_y, m2_z-m1_z])

    #calcul de la box autour de la bactérie
    x_min = bact.pos_x - bact.length/2.0 - bact.thickness/2.0
    x_max = bact.pos_x + bact.length/2.0 + bact.thickness/2.0
    y_min = bact.pos_y - bact.length/2.0 - bact.thickness/2.0
    y_max = bact.pos_y + bact.length/2.0 + bact.thickness/2.0
    z_min = bact.pos_z - bact.length/2.0 - bact.thickness/2.0
    z_max = bact.pos_z + bact.length/2.0 + bact.thickness/2.0

    #calcul des index correspondants
    i_x_min = int(x_min / vox_size_xy)
    i_x_max = int(x_max / vox_size_xy)
    i_y_min = int(y_min / vox_size_xy)
    i_y_max = int(y_max / vox_size_xy)
    i_z_min = int(z_min / vox_size_z)
    i_z_max = int(z_max / vox_size_z)

    i_x_min = max(0, i_x_min)
    i_x_max = min(i_x_max, mask_volume.shape[0])
    i_y_min = max(0, i_y_min)
    i_y_max = min(i_y_max, mask_volume.shape[1])
    i_z_min = max(0, i_z_min)
    i_z_max = min(i_z_max, mask_volume.shape[2])

    count = 0

    for x in range(i_x_min, i_x_max):
        for y in range(i_y_min, i_y_max):
            for z in range(i_z_min, i_z_max):

                #calcul de la position
                pos_x = x * vox_size_xy
                pos_y = y * vox_size_xy
                pos_z = z * vox_size_z

                vox_m1 = np.array([pos_x-m1_x, pos_y-m1_y, pos_z-m1_z])

                #calcul de la distance de la position xyz avec le segment [m1 m2]
                distance = np.linalg.norm(np.cross(m2m1, vox_m1))/ np.linalg.norm(m2m1)
                # print(distance)
                if (distance < bact.thickness/2.0):
                    mask_volume[x,y,z] = True




#######################################################################################################
#########################################           MAIN        #######################################
#######################################################################################################


if __name__ == "__main__":

    #Creation des repertoires d'enregistrement:
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    print("Date et heure actuelles:", formatted_date_time)

    chemin_positions = "./results/" + formatted_date_time + "/positions"
    chemin_holograms = "./results/" + formatted_date_time + "/holograms"

    os.makedirs(chemin_positions, exist_ok=True)
    os.makedirs(chemin_holograms, exist_ok=True)

    # définition du volume à simuler
    x_size = 512
    y_size = 512
    z_size = 1000

    volume_size = [x_size, y_size, z_size]

    transmission_milieu = 1.0
    index_milieu = 1.33
    index_bactérie = 1.5

    pitch = 5.5e-6
    vox_size_xy = pitch / 40
    vox_size_z = 100e-6 / z_size

    #parametres source illumination
    moyenne = 1.0
    ecart_type = 0.1
    bruit_gaussien = np.abs(np.random.normal(moyenne, ecart_type, [x_size, y_size]))
    
    wavelenght = 660e-9
    lambda_milieu = wavelenght / index_milieu
    shift_in_env = 0.0
    shift_in_obj = 2.0 * cp.pi * vox_size_z * (index_bactérie - index_milieu) / wavelenght

    #allocations
    h_holo = np.zeros(shape = (x_size, y_size), dtype = np.float32)
    d_holo = cp.zeros(shape = (x_size, y_size), dtype = cp.float32)
    d_fft_holo = cp.zeros(shape = (x_size, y_size), dtype = cp.complex64)
    d_fft_holo_propag = cp.zeros(shape = (x_size, y_size), dtype = cp.complex64)
    d_holo_propag = cp.zeros(shape = (x_size, y_size), dtype = cp.float32)
    d_KERNEL = cp.zeros(shape = (x_size, y_size), dtype = cp.complex64)

    nb_holo_to_simulate = 100
    number_of_bacteria = 25

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
                            thickness=1e-6, length=3e-6, transmission=1.0, ref_index=index_bactérie
                            )

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
                                                lambda_milieu, 40.0, pitch, x_size, y_size, vox_size_z, 0,0)
            
            maskplane = cp.asarray(mask_volume[:,:,i])

            field_plane = phase_shift_through_plane(mask_plane=maskplane, plane_to_shift=field_plane,
                                                    shift_in_env=shift_in_env, shift_in_obj=shift_in_obj)
            
        traitement_holo.save_image(traitement_holo.intensite(field_plane), chemin_holograms + "/holo_" + str(n) + ".bmp")

    # traitement_holo.affichage(traitement_holo.intensite(field_plane))
