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
import math
import matplotlib.pyplot as plt
import tifffile  

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

def insert_bact_in_mask_volume(mask_volume: np, bact: Bacterie, vox_size_xy: float, vox_size_z: float):
    
    phi_rad = math.radians(bact.phi)
    theta_rad = math.radians(bact.theta)
    x_size_upscaled = mask_volume.shape[0]
    y_size_upscaled = mask_volume.shape[1]

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
    i_x_min = int(x_min / vox_size_xy)
    i_x_max = int(math.ceil(x_max / vox_size_xy))
    i_y_min = int( y_min / vox_size_xy)
    i_y_max = int(math.ceil(y_max / vox_size_xy))
    i_z_min = int(z_min / vox_size_z)
    i_z_max = int(math.ceil(z_max / vox_size_z))

    i_x_min = max(0, i_x_min)
    i_x_max = min(i_x_max, x_size_upscaled)
    i_y_min = max(0, i_y_min)
    i_y_max = min(i_y_max, y_size_upscaled)
    i_z_min = max(0, i_z_min)
    i_z_max = min(i_z_max, mask_volume.shape[2])

    for z in range(i_z_min, i_z_max):
        for y in range(i_y_min, i_y_max):
            for x in range(i_x_min, i_x_max):

                # Position du voxel en mètres
                pos_x = x * vox_size_xy
                pos_y = y * vox_size_xy
                pos_z = z * vox_size_z

                P = np.array([pos_x, pos_y, pos_z])
                A = np.array([m1_x, m1_y, m1_z])
                B = np.array([m2_x, m2_y, m2_z])
                AB = B - A
                AP = P - A

                # projection scalaire (t entre 0 et 1 si à l'intérieur du segment)
                t = np.dot(AP, AB) / np.dot(AB, AB)

                if t < 0.0:
                    closest_point = A
                elif t > 1.0:
                    closest_point = B
                else:
                    closest_point = A + t * AB

                distance = np.linalg.norm(P - closest_point)

                if distance < bact.thickness / 2.0:
                    mask_volume[x, y, z] = 1.0

    return

@jit.rawkernel()
def insert_bact_kernel(mask_volume,
                       m1_x, m1_y, m1_z,
                       m2m1_x, m2m1_y, m2m1_z,
                       i_x_min, i_y_min, i_z_min,
                       vox_size_xy, vox_size_z,
                       threshold,
                       x_plane_size, y_plane_size, z_plane_size):
    
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    plane_size = x_plane_size * y_plane_size
    total = plane_size * z_plane_size

    if tid < total:
        # z = tid // plane_size
        # y = (total - (z * plane_size)) // x_plane_size
        # x = total - (z * plane_size) - y * x_plane_size

        z = tid // (plane_size)
        y = (tid % (plane_size)) // x_plane_size
        x = tid % x_plane_size

        pos_x = (i_x_min + x) * vox_size_xy
        pos_y = (i_y_min + y) * vox_size_xy
        pos_z = (i_z_min + z) * vox_size_z

        wx = pos_x - m1_x
        wy = pos_y - m1_y
        wz = pos_z - m1_z

        v_dot_v = m2m1_x * m2m1_x + m2m1_y * m2m1_y + m2m1_z * m2m1_z
        dot = wx * m2m1_x + wy * m2m1_y + wz * m2m1_z
        t = dot / v_dot_v

        if t < 0.0:
            # plus proche de m1
            dx = wx
            dy = wy
            dz = wz
        elif t > 1.0:
            # plus proche de m2
            dx = pos_x - (m1_x + m2m1_x)
            dy = pos_y - (m1_y + m2m1_y)
            dz = pos_z - (m1_z + m2m1_z)
        else:
            # projection sur le segment
            proj_x = m1_x + t * m2m1_x
            proj_y = m1_y + t * m2m1_y
            proj_z = m1_z + t * m2m1_z
            dx = pos_x - proj_x
            dy = pos_y - proj_y
            dz = pos_z - proj_z

        distance_squared = dx * dx + dy * dy + dz * dz

        if distance_squared < threshold * threshold:
            mask_volume[i_x_min + x, i_y_min + y, i_z_min + z] = 1.0

def GPU_insert_bact_in_mask_volume(mask_volume, bact, vox_size_xy, vox_size_z):
    phi = math.radians(bact.phi)
    theta = math.radians(bact.theta)

    long_half_seg = (bact.length - bact.thickness / 2.0) / 2.0

    m1_x = bact.pos_x - long_half_seg * math.sin(phi) * math.cos(theta)
    m1_y = bact.pos_y - long_half_seg * math.sin(phi) * math.sin(theta)
    m1_z = bact.pos_z - long_half_seg * math.cos(phi)

    m2_x = bact.pos_x + long_half_seg * math.sin(phi) * math.cos(theta)
    m2_y = bact.pos_y + long_half_seg * math.sin(phi) * math.sin(theta)
    m2_z = bact.pos_z + long_half_seg * math.cos(phi)

    m2m1_x = m2_x - m1_x
    m2m1_y = m2_y - m1_y
    m2m1_z = m2_z - m1_z

    x_min = bact.pos_x - bact.length / 2.0 - bact.thickness / 2.0
    x_max = bact.pos_x + bact.length / 2.0 + bact.thickness / 2.0
    y_min = bact.pos_y - bact.length / 2.0 - bact.thickness / 2.0
    y_max = bact.pos_y + bact.length / 2.0 + bact.thickness / 2.0
    z_min = bact.pos_z - bact.length / 2.0 - bact.thickness / 2.0
    z_max = bact.pos_z + bact.length / 2.0 + bact.thickness / 2.0

    i_x_min = max(0, int(x_min / vox_size_xy))
    i_x_max = min(int(math.ceil(x_max / vox_size_xy)), mask_volume.shape[0]-1)
    i_y_min = max(0, int(y_min / vox_size_xy))
    i_y_max = min(int(math.ceil(y_max / vox_size_xy)), mask_volume.shape[1]-1)
    i_z_min = max(0, int(z_min / vox_size_z))
    i_z_max = min(int(math.ceil(z_max / vox_size_z)), mask_volume.shape[2]-1)

    x_plane_size = i_x_max - i_x_min
    y_plane_size = i_y_max - i_y_min
    z_plane_size = i_z_max - i_z_min

    total_voxels = x_plane_size * y_plane_size * z_plane_size
    nthread = 1024
    nBlock = math.ceil(total_voxels // nthread) + 1

    insert_bact_kernel[nBlock, nthread](
        mask_volume,
        m1_x, m1_y, m1_z,
        m2m1_x, m2m1_y, m2m1_z,
        i_x_min, i_y_min, i_z_min,
        vox_size_xy, vox_size_z,
        bact.thickness * 0.5,
        x_plane_size, y_plane_size, z_plane_size
    )

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

def pad_centered(array, target_shape):
    """Pad 2D array to be centered in a target shape."""
    pad_x = target_shape[0] - array.shape[0]
    pad_y = target_shape[1] - array.shape[1]

    pad_x_before = pad_x // 2
    pad_x_after = pad_x - pad_x_before
    pad_y_before = pad_y // 2
    pad_y_after = pad_y - pad_y_before

    padded = cp.pad(array, ((pad_x_before, pad_x_after), (pad_y_before, pad_y_after)), mode='constant')
    return padded


def save_holo_data(filepath_npz, hologram_volume: np.ndarray,
                   hologram_image: np.ndarray, parameters: dict, bacteria_list: list[dict]):
    
    # Structuré : bactéries
    bacteria_dtype = [
        ('thickness', 'f4'), ('length', 'f4'),
        ('x_position_m', 'f4'), ('y_position_m', 'f4'), ('z_position_m', 'f4'),
        ('theta_angle', 'f4'), ('phi_angle', 'f4')
    ]
    bacteria_array = np.array([
        (
            b["thickness"], b["length"],
            b["x_position_m"], b["y_position_m"], b["z_position_m"],
            b["theta_angle"], b["phi_angle"]
        )
        for b in bacteria_list
    ], dtype=bacteria_dtype)

    # Sauvegarde .npz avec le volume en booléen
    np.savez(
        filepath_npz,
        hologram_volume=hologram_volume.astype(np.bool_),
        parameters=parameters,
        bacteria=bacteria_array,
        hologram_image=hologram_image.astype(np.float32)
    )


def load_holo_data(filepath_npz):
    with np.load(filepath_npz, allow_pickle=True) as npz:
        hologram_volume = npz["hologram_volume"].astype(np.bool_)  # Assure la cohérence
        parameters = npz["parameters"].item()
        bacteria_array = npz["bacteria"]
        hologram_image = npz["hologram_image"]

    bacteria_list = [
        {name: row[name] for name in bacteria_array.dtype.names}
        for row in bacteria_array
    ]

    return hologram_volume, hologram_image, parameters, bacteria_list

def save_volume_as_tiff(filepath_tiff, hologram_volume: np.ndarray):
    """
    Sauvegarde le volume 3D booléen en TIFF multi-stack visualisable.
    
    Args:
        filepath_tiff: Chemin du fichier TIFF (ex: "output/volume.tif")
        hologram_volume: Volume 3D booléen (X, Y, Z)
    """
    # Conversion en uint8 pour la visualisation (0 ou 255)
    volume_uint8 = (hologram_volume.astype(np.uint8) * 255)
    
    # Sauvegarde avec axe Z comme stack
    tifffile.imwrite(filepath_tiff, volume_uint8, photometric='minisblack')
    
    print(f"Volume 3D sauvegardé : {filepath_tiff}")
