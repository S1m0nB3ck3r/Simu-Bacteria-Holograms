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
import argparse

class Bacterie():

    def __init__(self):
        
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0
        self.thickness = 0.0
        self.length = 0.0
        self.theta = 0.0
        self.phi = 0.0

    def __init__(self, pos_x, pos_y, pos_z,
                 thickness, length,
                 theta, phi):
        
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.thickness = thickness
        self.length = length
        self.theta = theta
        self.phi = phi

    def to_file(self, path_file):

        txt = "{posx}\t{posy}\t{posz}\t{angle1}\t{angle2}\t\n".format(posx = self.pos_x, posy = self.pos_y, posz = self.pos_z, angle1 = self.theta, angle2 = self.phi)

        with open(path_file, "a") as file:
            file.write(txt)



class Sphere():

    def __init__(self):
        
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0
        self.radius = 0.0

    def __init__(self, pos_x, pos_y, pos_z,
                 radius):
        
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.radius = radius

    def to_file(self, path_file):

        txt = "{posx}\t{posy}\t{posz}\t{radius}\t\n".format(posx = self.pos_x, posy = self.pos_y, posz = self.pos_z, radius = self.radius)

        with open(path_file, "a") as file:
            file.write(txt)




def gen_random_bacteria(number_of_bact: int, xyz_min_max: list, thickness: float, length: float):

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
                                theta_angles[i], phi_angles[i]))
        
        return list_bact

def gen_random_sphere(number_of_sphere: int, xyz_min_max: list, radius: float):

        rng = np.random.default_rng()

        x_min, x_max, y_min, y_max, z_min, z_max = xyz_min_max

        list_bact = []

        x_positions = (x_max - x_min) * rng.random(number_of_sphere) + x_min
        y_positions = (y_max - y_min) * rng.random(number_of_sphere) + y_min
        z_positions = (z_max - z_min) * rng.random(number_of_sphere) + z_min
        
        for i in range(number_of_sphere):

            list_bact.append(Sphere(x_positions[i], y_positions[i], z_positions[i], radius=radius))
        
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

def cross_through_plane(mask_plane :cp, plane_to_shift: cp, shift_in_env: float, shift_in_obj: float, transmission_in_obj: float):

    shift_plane = cp.full(fill_value=shift_in_env, dtype=cp.float32, shape=mask_plane.shape)
    transmission_plane = cp.full(fill_value=1.0, dtype=cp.float32, shape=mask_plane.shape)

    cp.putmask(a=shift_plane, mask=mask_plane, values=shift_in_obj)
    cp.putmask(a=transmission_plane, mask=mask_plane, values=transmission_in_obj)

    def phase_shift(cplx_plane, shift_plane, transmission_plane):

        phase = cp.angle(cplx_plane)
        module = cp.sqrt(cp.real(cplx_plane) ** 2 + cp.imag(cplx_plane) ** 2)

        phase = phase + shift_plane

        return module* transmission_plane * cp.exp((0+1.j) * phase)
    
    return phase_shift(plane_to_shift, shift_plane, transmission_plane)

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


def insert_sphere_in_mask_volume(mask_volume: np, sphere: Sphere, vox_size_xy: float, vox_size_z: float):

    #calcul de la box autour de la sphere
    x_min = sphere.pos_x  - sphere.radius
    x_max = sphere.pos_x  + sphere.radius
    y_min = sphere.pos_y  - sphere.radius
    y_max = sphere.pos_y  + sphere.radius
    z_min = sphere.pos_z  - sphere.radius
    z_max = sphere.pos_z  + sphere.radius

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

                #calcul de la distance de la position de la sphere avec le voxel
                distance = np.sqrt((pos_x - sphere.pos_x)**2 + (pos_y - sphere.pos_y)**2 + (pos_z - sphere.pos_z)**2)
                # print(distance)
                if (distance < sphere.radius):
                    mask_volume[x,y,z] = True