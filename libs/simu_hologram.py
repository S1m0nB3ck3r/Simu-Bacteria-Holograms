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
from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
from scipy.ndimage import distance_transform_edt as scipy_distance_transform_edt
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

def phase_shift_through_plane(mask_plane :cp, plane_to_shift: cp, shift_in_env: float, shift_in_obj: float):

    # Interpolation linéaire : masque lissé (0→1) donne un déphasage gradué
    # Pour masque binaire : résultat identique à l'ancien code
    # Pour masque lissé  : transition douce du déphasage (anti-aliasing)
    mask_f32 = mask_plane.astype(cp.float32)
    shift_plane = mask_f32 * shift_in_obj

    return plane_to_shift * cp.exp(1j * shift_plane)


def cross_through_plane(mask_plane :cp, plane_to_shift: cp, shift_in_env: float, shift_in_obj: float, transmission_in_obj: float):

    # Interpolation linéaire basée sur la valeur du masque (0→1)
    # Pour masque binaire : résultat identique à l'ancien code
    # Pour masque lissé  : transition douce (anti-aliasing)
    mask_f32 = mask_plane.astype(cp.float32)
    shift_plane = mask_f32 * shift_in_obj
    transmission_plane = 1.0 - (mask_f32 * transmission_in_obj)

    return plane_to_shift * transmission_plane * cp.exp(1j * shift_plane)

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

def insert_sphere_in_mask_volume(mask_volume, sphere: Sphere, vox_size_xy: float, vox_size_z: float, upscale_factor: int = 1):
    """
    Insère une sphère dans le volume masque fourni.

    Convention identique à GPU_insert_bact_in_mask_volume :
      - mask_volume est le volume dans lequel on écrit (déjà sur-échantillonné
        en XY si l'appelant le souhaite). Il ne contient PAS de border : le
        cadre anti-aliasing est ajouté plus tard, plan par plan, par
        pad_centered() au moment de la propagation, puis retiré au recadrage.
      - vox_size_xy / vox_size_z sont les tailles de voxel DE CE VOLUME.
      - les coordonnées de la sphère sont exprimées dans ce même repère,
        c'est-à-dire dans le volume central.

    upscale_factor n'intervient plus dans le calcul des index : c'est à
    l'appelant de fournir la taille de voxel correspondant au volume qu'il
    passe (comme le fait déjà le chemin bactéries). Le paramètre est conservé
    pour compatibilité d'appel.

    Fonctionne indifféremment sur un tableau NumPy ou CuPy.
    """
    xp = cp.get_array_module(mask_volume)

    x_size, y_size, z_size = mask_volume.shape

    # Box englobante de la sphère, en index de voxels
    i_x_min = max(0,      int(math.floor((sphere.pos_x - sphere.radius) / vox_size_xy)))
    i_x_max = min(x_size, int(math.ceil( (sphere.pos_x + sphere.radius) / vox_size_xy)) + 1)
    i_y_min = max(0,      int(math.floor((sphere.pos_y - sphere.radius) / vox_size_xy)))
    i_y_max = min(y_size, int(math.ceil( (sphere.pos_y + sphere.radius) / vox_size_xy)) + 1)
    i_z_min = max(0,      int(math.floor((sphere.pos_z - sphere.radius) / vox_size_z)))
    i_z_max = min(z_size, int(math.ceil( (sphere.pos_z + sphere.radius) / vox_size_z)) + 1)

    # Sphère entièrement hors du volume
    if i_x_min >= i_x_max or i_y_min >= i_y_max or i_z_min >= i_z_max:
        return mask_volume

    # Distances au centre, calculées sur la seule box englobante
    dx = xp.arange(i_x_min, i_x_max, dtype=xp.float32) * vox_size_xy - sphere.pos_x
    dy = xp.arange(i_y_min, i_y_max, dtype=xp.float32) * vox_size_xy - sphere.pos_y
    dz = xp.arange(i_z_min, i_z_max, dtype=xp.float32) * vox_size_z  - sphere.pos_z

    inside = (dx[:, None, None] ** 2
              + dy[None, :, None] ** 2
              + dz[None, None, :] ** 2) < (sphere.radius ** 2)

    # maximum() et non affectation directe : ne pas effacer les sphères déjà insérées
    sub = (slice(i_x_min, i_x_max), slice(i_y_min, i_y_max), slice(i_z_min, i_z_max))
    mask_volume[sub] = xp.maximum(mask_volume[sub], inside.astype(mask_volume.dtype))

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


def create_illumination_field(field_size_xy_pix: int, wavelength: float, pixel_size: float, medium_index: float,
                              magnification: float,
                              number_of_sources: int, sources_angle_degree_X: list[float], sources_angle_degree_Y: list[float],
                              noise_mean: float = 1.0, noise_std: float = 0.05) -> cp.ndarray:
    """
    Crée un champ d'illumination 2D multi-sources avec bruit gaussien sur l'amplitude
    et phase de départ aléatoire.

    Le champ est échantillonné au plan objet avec un pas de pixel_size / magnification,
    cohérent avec propag_angular_spectrum.

    Args:
        field_size_xy_pix : taille du champ en pixels (carré)
        wavelength        : longueur d'onde dans le vide (m)
        pixel_size        : taille d'un pixel capteur (m)
        medium_index      : indice du milieu
        magnification     : grossissement du système optique
        number_of_sources : nombre de sources (ondes planes)
        sources_angle_degree_X : angles d'inclinaison selon X (degrés)
        sources_angle_degree_Y : angles d'inclinaison selon Y (degrés)
        noise_mean        : moyenne du bruit gaussien sur l'amplitude (défaut=1.0)
        noise_std         : écart-type du bruit gaussien sur l'amplitude (défaut=0.05)

    Returns:
        cp.ndarray: champ complexe 2D (complex64)
    """

    # Pas de pixel au plan objet (cohérent avec propag_angular_spectrum)
    effective_pixel_size = pixel_size / magnification

    center = field_size_xy_pix // 2
    x = (center - cp.arange(field_size_xy_pix)) * effective_pixel_size
    y = (center - cp.arange(field_size_xy_pix)) * effective_pixel_size
    X, Y = cp.meshgrid(x, y, indexing='ij')

    # Vecteur d'onde dans le milieu
    lambda_medium = wavelength / medium_index
    k0 = 2.0 * math.pi / lambda_medium

    field = cp.zeros((field_size_xy_pix, field_size_xy_pix), dtype=cp.complex64)

    for i in range(number_of_sources):
        angle_X_rad = math.radians(sources_angle_degree_X[i])
        angle_Y_rad = math.radians(sources_angle_degree_Y[i])
        kx = k0 * math.sin(angle_X_rad)
        ky = k0 * math.sin(angle_Y_rad)

        # Phase de départ aléatoire
        random_phase = 2 * math.pi * np.random.random()

        plane_wave = cp.exp(1j * (kx * X + ky * Y + random_phase))
        field += plane_wave

    # Normalisation par le nombre de sources
    field /= number_of_sources

    # Bruit gaussien sur l'amplitude (multiplication directe, pas de séparation module/phase)
    amplitude_noise = cp.asarray(
        np.abs(np.random.normal(noise_mean, noise_std, (field_size_xy_pix, field_size_xy_pix)))
    ).astype(cp.float32)

    field = field * amplitude_noise

    return field


def create_illumination_field_polar(
    field_size_xy_pix: int,
    wavelength: float,
    pixel_size: float,
    medium_index: float,
    magnification: float,
    number_of_sources: int,
    sources_polar_degree: list[float],   # angle polaire
    sources_azimuth_degree: list[float],     # azimut
    noise_mean: float = 1.0,
    noise_std: float = 0.05
) -> cp.ndarray:

    effective_pixel_size = pixel_size / magnification

    center = field_size_xy_pix // 2
    x = (center - cp.arange(field_size_xy_pix)) * effective_pixel_size
    y = (center - cp.arange(field_size_xy_pix)) * effective_pixel_size
    X, Y = cp.meshgrid(x, y, indexing='ij')

    lambda_medium = wavelength / medium_index
    k0 = 2.0 * math.pi / lambda_medium

    field = cp.zeros((field_size_xy_pix, field_size_xy_pix), dtype=cp.complex64)

    for i in range(number_of_sources):

        polar_rad = math.radians(sources_polar_degree[i])
        azimuth_rad = math.radians(sources_azimuth_degree[i])

        # Décomposition correcte du vecteur d’onde
        kx = k0 * math.sin(polar_rad) * math.cos(azimuth_rad)
        ky = k0 * math.sin(polar_rad) * math.sin(azimuth_rad)

        random_phase = 2 * math.pi * np.random.random()

        plane_wave = cp.exp(1j * (kx * X + ky * Y + random_phase))
        field += plane_wave

    field /= number_of_sources

    amplitude_noise = cp.asarray(
        np.abs(np.random.normal(noise_mean, noise_std,
        (field_size_xy_pix, field_size_xy_pix)))
    ).astype(cp.float32)

    field *= amplitude_noise

    return field

def smooth_volume_gaussian_gpu(cp_mask_volume, sigma=0.5):
    """
    Lisse un volume CuPy 3D par filtre gaussien sur GPU.
    Réduit l'aliasing des bords voxelisés lors de la propagation angulaire.

    Args:
        cp_mask_volume: volume CuPy 3D (float16 ou float32)
        sigma: écart-type du filtre gaussien en voxels (défaut=0.5)
               Plus sigma est grand, plus le lissage est fort.
               Valeurs recommandées : 0.3 (angle faible) à 0.8 (angle fort)

    Returns:
        volume lissé CuPy (float32)
    """
    vol = cp_mask_volume.astype(cp.float32)
    smoothed = cp_gaussian_filter(vol, sigma=sigma)
    # Clamp entre 0 et 1 pour rester cohérent avec un masque
    smoothed = cp.clip(smoothed, 0.0, 1.0)
    print(f"    [Smoothing] Gaussien GPU appliqué (sigma={sigma})")
    return smoothed


def smooth_volume_sdf_gpu(cp_mask_volume, sdf_width=1.0):
    """
    Lisse un volume CuPy 3D par Signed Distance Field (SDF).
    Préserve mieux la géométrie que le filtre gaussien.
    Le calcul de la distance se fait sur CPU (scipy), puis la sigmoïde sur GPU.

    Args:
        cp_mask_volume: volume CuPy 3D (float16 ou float32)
        sdf_width: largeur de la transition douce en voxels (défaut=1.0)
                   Plus sdf_width est grand, plus la transition est douce.
                   Valeurs recommandées : 0.5 (peu de lissage) à 2.0 (fort lissage)

    Returns:
        volume lissé CuPy (float32)
    """
    # Transfert CPU pour distance_transform_edt (non dispo sur GPU)
    vol_np = cp.asnumpy(cp_mask_volume.astype(cp.float32))
    binary = vol_np > 0.5

    dist_inside = scipy_distance_transform_edt(binary)
    dist_outside = scipy_distance_transform_edt(~binary)

    # SDF signé : positif à l'intérieur, négatif à l'extérieur
    sdf_np = dist_inside - dist_outside

    # Retour GPU pour la sigmoïde
    sdf_gpu = cp.asarray(sdf_np, dtype=cp.float32)
    smoothed = 1.0 / (1.0 + cp.exp(-sdf_gpu / sdf_width))

    print(f"    [Smoothing] SDF appliqué (sdf_width={sdf_width})")
    return smoothed


def smooth_volume_gpu(cp_mask_volume, method='gaussian', sigma=0.5, sdf_width=1.0):
    """
    Lisse un volume CuPy 3D pour réduire l'aliasing des bords voxelisés.
    Dispatche vers la méthode choisie (gaussien GPU ou SDF).

    Args:
        cp_mask_volume: volume CuPy 3D (float16 ou float32)
        method: 'gaussian' ou 'sdf'
        sigma: paramètre du filtre gaussien (voxels)
        sdf_width: largeur de transition SDF (voxels)

    Returns:
        volume lissé CuPy (float32)
    """
    if method == 'gaussian':
        return smooth_volume_gaussian_gpu(cp_mask_volume, sigma=sigma)
    elif method == 'sdf':
        return smooth_volume_sdf_gpu(cp_mask_volume, sdf_width=sdf_width)
    else:
        raise ValueError(f"Méthode de lissage inconnue : '{method}'. Utiliser 'gaussian' ou 'sdf'.")


def display_complex_plane(complex_plane, title: str = ""):
    """
    Affiche le module et la phase d'un plan 2D complexe côte à côte.

    Args:
        complex_plane: plan 2D complexe (cp.ndarray ou np.ndarray)
        title: titre optionnel pour la figure
    """
    if isinstance(complex_plane, cp.ndarray):
        complex_plane = cp.asnumpy(complex_plane)

    module = np.abs(complex_plane)
    phase = np.angle(complex_plane)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(module, cmap='gray')
    ax1.set_title("Module")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(phase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax2.set_title("Phase")
    plt.colorbar(im2, ax=ax2)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()


def display_real_plane(real_plane, title: str = "", cmap: str = "gray"):
    """
    Affiche une image d'un plan 2D réel.

    Args:
        real_plane: plan 2D réel (cp.ndarray ou np.ndarray)
        title: titre optionnel pour la figure
        cmap: colormap matplotlib (défaut='gray')
    """
    if isinstance(real_plane, cp.ndarray):
        real_plane = cp.asnumpy(real_plane)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    im = ax.imshow(real_plane, cmap=cmap)
    if title:
        ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()
