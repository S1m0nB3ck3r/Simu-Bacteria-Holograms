# -*- coding: utf-8 -*-

"""
Filename: simu_hologram.py

Description:
Functions needed to generate a virtual volume with objects (spheres and bacteria) includeed in order to create synthetic holograms.
Author: Simon BECKER
Date: 2024-07-09

License:
GNU General Public License v3.0

Copyright (C) [2024] Simon BECKER

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

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

        txt = "{posx}\t{posy}\t{posz}\t{lengh}\t{thickness}\t{angle1}\t{angle2}\t\n".format(
            posx = self.pos_x, posy = self.pos_y, posz = self.pos_z, lengh = self.length,
            thickness = self.thickness, angle1 = self.theta, angle2 = self.phi
            )

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

def gen_random_bacteria(number_of_bact: int, xyz_min_max: list, thickness_min_max: dict, length_min_max: dict):

        rng = np.random.default_rng()

        x_min, x_max, y_min, y_max, z_min, z_max = xyz_min_max

        list_bact = []
        thickness = (thickness_min_max[max] - thickness_min_max[min]) * rng.random(number_of_bact) + thickness_min_max[min]
        length = (length_min_max[max] - length_min_max[min]) * rng.random(number_of_bact) + length_min_max[min]
        x_positions = (x_max - x_min) * rng.random(number_of_bact) + x_min
        y_positions = (y_max - y_min) * rng.random(number_of_bact) + y_min
        z_positions = (z_max - z_min) * rng.random(number_of_bact) + z_min

        theta_angles = 90.0 * rng.random(number_of_bact)
        phi_angles = 90.0 * rng.random(number_of_bact)
        
        for i in range(number_of_bact):

            list_bact.append(Bacterie(x_positions[i], y_positions[i], z_positions[i], thickness[i], length[i],
                                theta_angles[i], phi_angles[i]))
        
        return list_bact

def gen_random_sphere(number_of_sphere: int, xyz_min_max: list, radius_min_max: dict):

        rng = np.random.default_rng()

        x_min, x_max, y_min, y_max, z_min, z_max = xyz_min_max

        list_bact = []
        radius = (radius_min_max[max] - radius_min_max[min]) * rng.random(number_of_sphere) + radius_min_max[min]
        x_positions = (x_max - x_min) * rng.random(number_of_sphere) + x_min
        y_positions = (y_max - y_min) * rng.random(number_of_sphere) + y_min
        z_positions = (z_max - z_min) * rng.random(number_of_sphere) + z_min
        
        for i in range(number_of_sphere):

            list_bact.append(Sphere(x_positions[i], y_positions[i], z_positions[i], radius=radius[i]))
        
        return list_bact

def phase_correction(cplx_plane, shift_plane):

    phase = cp.angle(cplx_plane)
    module = cp.sqrt(cp.real(cplx_plane) ** 2 + cp.imag(cplx_plane) ** 2)

    phase = phase + shift_plane

    return module * cp.exp((0+1.j) * phase)

def phase_shift_through_plane(mask_plane :cp, plane_to_shift: cp, shift_in_env: float, shift_in_obj: float):

    # shift_plane = cp.full(fill_value=shift_in_env, dtype=cp.float32, shape=mask_plane.shape)
    shift_plane = mask_plane * shift_in_obj

    cp.putmask(a=shift_plane, mask=mask_plane, values=shift_in_obj)

    return phase_correction(plane_to_shift, shift_plane)

def attenuation_and_phase_correction(d_cplx_plane, d_shift_plane, d_transmission_plane):

    phase_plane = cp.angle(d_cplx_plane)
    module = cp.sqrt(cp.real(d_cplx_plane) ** 2 + cp.imag(d_cplx_plane) ** 2)

    phase_plane = phase_plane + d_shift_plane

    return module* d_transmission_plane * cp.exp((0+1.j) * phase_plane)

def cross_through_plane(mask_plane :cp, plane_to_shift: cp, shift_in_env: float, shift_in_obj: float, transmission_in_obj: float):

    shift_plane = cp.full(fill_value=shift_in_env, dtype=cp.float32, shape=mask_plane.shape)
    transmission_plane = cp.full(fill_value=1.0, dtype=cp.float32, shape=mask_plane.shape)

    cp.putmask(a=shift_plane, mask=mask_plane, values=shift_in_obj)
    cp.putmask(a=transmission_plane, mask=mask_plane, values=transmission_in_obj)

    return attenuation_and_phase_correction(plane_to_shift, shift_plane, transmission_plane)

def insert_bact_in_mask_volume(mask_volume: np, bact: Bacterie, vox_size_xy: float, vox_size_z: float, upscale_factor: int = 1):
    
    phi_rad = math.radians(bact.phi)
    theta_rad = math.radians(bact.theta)
    x_size_upscaled = mask_volume.shape[0]*upscale_factor
    y_size_upscaled = mask_volume.shape[1]*upscale_factor

    #distance Extremité-centre (en m):
    long_Demi_Seg = (bact.length - bact.thickness/2.0) / 2.0

	#calcul des positions des extremités du segment interieur de la bactérie m1 et m2 (positions en m)
    m1_x = bact.pos_x - long_Demi_Seg * math.sin(phi_rad) * math.cos(theta_rad)
    m1_y = bact.pos_y - long_Demi_Seg * math.sin(phi_rad) * math.sin(theta_rad)
    m1_z = bact.pos_z - long_Demi_Seg * math.cos(phi_rad)

    m2_x = bact.pos_x + long_Demi_Seg * math.sin(phi_rad) * math.cos(theta_rad)
    m2_y = bact.pos_y + long_Demi_Seg * math.sin(phi_rad) * math.sin(theta_rad)
    m2_z = bact.pos_z + long_Demi_Seg * math.cos(phi_rad)

    #calcul segment [m2 m1]
    m2m1 = np.array([m2_x-m1_x, m2_y-m1_y, m2_z-m1_z])

    #calcul de la box autour de la bactérie (positions en m)
    x_min = bact.pos_x - bact.length/2.0 - bact.thickness/2.0
    x_max = bact.pos_x + bact.length/2.0 + bact.thickness/2.0
    y_min = bact.pos_y - bact.length/2.0 - bact.thickness/2.0
    y_max = bact.pos_y + bact.length/2.0 + bact.thickness/2.0
    z_min = bact.pos_z - bact.length/2.0 - bact.thickness/2.0
    z_max = bact.pos_z + bact.length/2.0 + bact.thickness/2.0

    #calcul des index correspondants
    i_x_min = int(upscale_factor * x_min / vox_size_xy)
    i_x_max = int(math.ceil(upscale_factor * x_max / vox_size_xy))
    i_y_min = int(upscale_factor * y_min / vox_size_xy)
    i_y_max = int(math.ceil(upscale_factor * y_max / vox_size_xy))
    i_z_min = int(z_min / vox_size_z)
    i_z_max = int(math.ceil(z_max / vox_size_z))

    i_x_min = max(0, i_x_min)
    i_x_max = min(i_x_max, x_size_upscaled)
    i_y_min = max(0, i_y_min)
    i_y_max = min(i_y_max, y_size_upscaled)
    i_z_min = max(0, i_z_min)
    i_z_max = min(i_z_max, mask_volume.shape[2])

    for z in range(i_z_min, i_z_max):

        plane = np.zeros(dtype = np.float16, shape= (x_size_upscaled, y_size_upscaled))
        for y in range(i_y_min, i_y_max):
            for x in range(i_x_min, i_x_max):

                #calcul de la position en µm
                pos_x = x * vox_size_xy / upscale_factor
                pos_y = y * vox_size_xy / upscale_factor
                pos_z = z * vox_size_z

                vox_m1 = np.array([pos_x-m1_x, pos_y-m1_y, pos_z-m1_z])

                #calcul de la distance de la position xyz avec le segment [m1 m2]
                distance = np.linalg.norm(np.cross(m2m1, vox_m1))/ np.linalg.norm(m2m1)
                # print(distance)
                if (distance < bact.thickness/2.0):
                    plane[x,y] = 1.0
        
        plane = plane.reshape(mask_volume.shape[0], upscale_factor, mask_volume.shape[1], upscale_factor)

        mask_volume[:,:,z] = plane.mean(axis = (1,3))

    return mask_volume

def insert_sphere_in_mask_volume(mask_volume: np, sphere: Sphere, vox_size_xy: float, vox_size_z: float, upscale_factor: int = 1):

    x_size_upscaled = mask_volume.shape[0]*upscale_factor
    y_size_upscaled = mask_volume.shape[1]*upscale_factor

    #calcul de la box autour de la sphere (en m)
    x_min = sphere.pos_x  - sphere.radius
    x_max = sphere.pos_x  + sphere.radius
    y_min = sphere.pos_y  - sphere.radius
    y_max = sphere.pos_y  + sphere.radius
    z_min = sphere.pos_z  - sphere.radius
    z_max = sphere.pos_z  + sphere.radius

    #calcul des index correspondants 
    i_x_min = int(upscale_factor * x_min / vox_size_xy)
    i_x_max = int(math.ceil(upscale_factor * x_max / vox_size_xy))
    i_y_min = int(upscale_factor * y_min / vox_size_xy)
    i_y_max = int(math.ceil(upscale_factor * y_max / vox_size_xy))
    i_z_min = int(z_min / vox_size_z)
    i_z_max = int(math.ceil(z_max / vox_size_z))

    i_x_min = max(0, i_x_min)
    i_x_max = min(i_x_max, x_size_upscaled)
    i_y_min = max(0, i_y_min)
    i_y_max = min(i_y_max, y_size_upscaled)
    i_z_min = max(0, i_z_min)
    i_z_max = min(i_z_max, mask_volume.shape[2])

    for z in range(i_z_min, i_z_max):

        plane = np.zeros(dtype = np.float16, shape= (x_size_upscaled, y_size_upscaled))

        for x in range(i_x_min, i_x_max):
            for y in range(i_y_min, i_y_max):

                #calcul de la position
                pos_x = x * vox_size_xy / upscale_factor
                pos_y = y * vox_size_xy / upscale_factor
                pos_z = z * vox_size_z

                #calcul de la distance de la position de la sphere avec le voxel
                distance = np.sqrt((pos_x - sphere.pos_x)**2 + (pos_y - sphere.pos_y)**2 + (pos_z - sphere.pos_z)**2)
                # print(distance)
                if (distance < sphere.radius):
                    plane[x,y] = 1.0

        plane = plane.reshape(mask_volume.shape[0], upscale_factor, mask_volume.shape[1], upscale_factor)
        mask_volume[:,:,z] = plane.mean(axis = (1,3))

    return mask_volume