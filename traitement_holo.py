# -*- coding: utf-8 -*-

"""
Filename: traitement_holo.py

Description:
different kind of treatments needeed to hologram analysis or display
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
import os
from PIL import Image
import numpy as np
import cupy as cp
import time
import math
import cc3d
from cupyx import jit
import cupy as cp
from cupy.fft import rfft2, fft2, ifft2, fftshift, ifftshift, fftn, ifftn
import matplotlib.pyplot as plt



def read_image(path_image, sizeX = 0, sizeY = 0):
        
        h_holo = np.asarray(Image.open(path_image))

        if ((sizeX != 0) and (sizeY != 0)):

            sx = np.size(h_holo, axis = 1)
            sy = np.size(h_holo, axis = 0)

            offsetX = (sx - sizeX)//2
            offsetY = (sy - sizeY)//2

            h_holo = h_holo[offsetY:offsetY+sizeY:1, offsetX:offsetX+sizeX:1]
        
        h_holo = h_holo.astype('float32')
        return(h_holo)

def save_image(image_array, path_image):

    if isinstance(image_array, cp.ndarray):
        h_image_array = cp.asnumpy(image_array)
        
    else:
        h_image_array = image_array

    min = h_image_array.min()
    max = h_image_array.max()

    h_image_array = ((h_image_array - min) * 255 / (max - min)).astype(np.uint8) 
    img = Image.fromarray(h_image_array)
    
    img.save(path_image)


def display(plan, title="Image"):
    """Fonction d'affichage robuste avec titre"""
    print(f"   🖼️  Affichage: {title}")
    
    if isinstance(plan, cp.ndarray):
        h_plan = cp.asnumpy(plan).astype(np.float32)
    else:
        h_plan = plan.astype(np.float32)
    
    # Utilise matplotlib pour un affichage avec titre garanti
    plt.figure(figsize=(8, 6))
    plt.imshow(h_plan, cmap='gray')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar(label='Intensité')
    plt.axis('off')  # Cache les axes pour un affichage plus propre
        
    # Affiche dans une fenêtre séparée
    plt.show(block=False)  # Non-bloquant pour permettre l'affichage de plusieurs images


@cp.fuse()
def div_holo(A, B):
    if (B!=0.0):
        C = A/B
    else:
        C = 0.0
    return C

def module(planComplex):
    if isinstance(planComplex, cp.ndarray):
        return(cp.sqrt(cp.square(cp.real(planComplex)) + cp.square(cp.imag(planComplex))))
    else:
        return(np.sqrt(np.square(np.real(planComplex)) + np.square(np.imag(planComplex))))

def intensite(planComplex):
    if isinstance(planComplex, cp.ndarray):
        return(cp.square(cp.real(planComplex)) + cp.square(cp.imag(planComplex)))
    else:
        return(np.square(np.real(planComplex)) + np.square(np.imag(planComplex)))

def phase(planComplex):
    if isinstance(planComplex, cp.ndarray):
        return(cp.arctan(cp.imag(planComplex) /cp.real(planComplex)))
    else:
        return(np.arctan(np.imag(planComplex) /np.real(planComplex)))

def affiche_particule(x, y, z, boxSizeXY, boxSizeZ, volume):

    sizeX, sizeY, sizeZ = volume.shape
    planXY = np.zeros(shape=(boxSizeXY, boxSizeXY))
    planXZ = np.zeros(shape=(boxSizeXY, boxSizeZ))
    planYZ = np.zeros(shape=(boxSizeXY, boxSizeZ))


    #test des limites des coordonées xyz
    xMin = int(x - boxSizeXY//2)
    xMax = int(x + boxSizeXY//2)
    yMin = int(y - boxSizeXY//2)
    yMax = int(y + boxSizeXY//2)
    zMin = int(z - boxSizeZ//2)
    zMax = int(z + boxSizeZ//2)

    xMin = xMin if xMin > 0 else 0
    xMax = xMax if xMax < sizeX else sizeX 
    yMin = yMin if yMin > 0 else 0 
    yMax = yMax if yMax < sizeY else sizeY
    zMin = zMin if zMin > 0 else 0 
    zMax = zMax if zMax < sizeZ else sizeZ

    if isinstance(volume, cp.ndarray):
        if (volume.dtype == cp.complex64):
            planXY_t = cp.asnumpy(intensite(volume[xMin : xMax, yMin : yMax, z ]))
            planXY[0:boxSizeXY, 0:boxSizeXY] = planXY_t
            planXZ[0:boxSizeXY, 0:boxSizeZ] = cp.asnumpy(intensite(volume[xMin : xMax, y, zMin : zMax]))
            planYZ[0:boxSizeXY, 0:boxSizeZ] = cp.asnumpy(intensite(volume[x , yMin : yMax, zMin : zMax ]))
        else:
            planXY[0:boxSizeXY, 0:boxSizeXY]  = cp.asnumpy(volume[xMin : xMax, yMin : yMax, z ])
            planXZ[0:boxSizeXY, 0:boxSizeZ]  = cp.asnumpy(volume[xMin : xMax, y, zMin : zMax])
            planYZ[0:boxSizeXY, 0:boxSizeZ]  = cp.asnumpy(volume[x , yMin : yMax, zMin : zMax ])
    else:
        if (volume.dtype == np.complex64):
            planXY[0:boxSizeXY, 0:boxSizeXY]  = intensite(volume[xMin : xMax, yMin : yMax, z ])
            planXZ[0:boxSizeXY, 0:boxSizeZ]  = intensite(volume[xMin : xMax, y, zMin : zMax])
            planYZ[0:boxSizeXY, 0:boxSizeZ]  = intensite(volume[x , yMin : yMax, zMin : zMax ])
        else:
            planXY[0:boxSizeXY, 0:boxSizeXY]  = volume[xMin : xMax, yMin : yMax, z ]
            planXZ[0:boxSizeXY, 0:boxSizeZ]  = volume[xMin : xMax, y, zMin : zMax]
            planYZ[0:boxSizeXY, 0:boxSizeZ]  = volume[x , yMin : yMax, zMin : zMax ]

    min = planXY.min()
    max = planXY.max()
    planXY = (planXY - min) * 255 / (max - min)
            
    #planXZ = np.rot90(planXZ)
    min = planXZ.min()
    max = planXZ.max()
    planXZ = (planXZ - min) * 255 / (max - min)
    #planXZ = np.rot90(planXZ)

            
    #planYZ = np.rot90(planYZ)
    min = planYZ.min()
    max = planYZ.max()
    planYZ = (planYZ - min) * 255 / (max - min)
    #planYZ = planYZ.astype(np.uint8)
    #planYZ = np.rot90(planYZ)
    planYZ.reshape((boxSizeXY, boxSizeZ))


    planTot = np.concatenate((planXY, planXZ, planYZ), axis = 1)
    img = Image.fromarray(planTot)
    img.show(title = "objet 3 plans")


def get_sub_plane(x, y, z, boxSizeXY, boxSizeZ, d_volume):

    sizeX, sizeY, sizeZ = d_volume.shape
    planXY = np.zeros(shape=(boxSizeXY, boxSizeXY))
    planXZ = np.zeros(shape=(boxSizeXY, boxSizeZ))
    planYZ = np.zeros(shape=(boxSizeXY, boxSizeZ))


    #test des limites des coordonées xyz
    xMin = int(x - boxSizeXY//2)
    xMax = int(x + boxSizeXY//2)
    yMin = int(y - boxSizeXY//2)
    yMax = int(y + boxSizeXY//2)
    zMin = int(z - boxSizeZ//2)
    zMax = int(z + boxSizeZ//2)

    xMin = xMin if xMin > 0 else 0
    xMax = xMax if xMax < sizeX else sizeX 
    yMin = yMin if yMin > 0 else 0 
    yMax = yMax if yMax < sizeY else sizeY
    zMin = zMin if zMin > 0 else 0 
    zMax = zMax if zMax < sizeZ else sizeZ

    if isinstance(d_volume, cp.ndarray):
        if (d_volume.dtype == cp.complex64):
            planXY_t = cp.asnumpy(intensite(d_volume[xMin : xMax, yMin : yMax, z ]))
            planXY[0:boxSizeXY, 0:boxSizeXY] = planXY_t
            planXZ[0:boxSizeXY, 0:boxSizeZ] = cp.asnumpy(intensite(d_volume[xMin : xMax, y, zMin : zMax]))
            planYZ[0:boxSizeXY, 0:boxSizeZ] = cp.asnumpy(intensite(d_volume[x , yMin : yMax, zMin : zMax ]))
        else:
            planXY[0:boxSizeXY, 0:boxSizeXY]  = cp.asnumpy(d_volume[xMin : xMax, yMin : yMax, z ])
            planXZ[0:boxSizeXY, 0:boxSizeZ]  = cp.asnumpy(d_volume[xMin : xMax, y, zMin : zMax])
            planYZ[0:boxSizeXY, 0:boxSizeZ]  = cp.asnumpy(d_volume[x , yMin : yMax, zMin : zMax ])
    else:
        if (d_volume.dtype == np.complex64):
            planXY[0:boxSizeXY, 0:boxSizeXY]  = intensite(d_volume[xMin : xMax, yMin : yMax, z ])
            planXZ[0:boxSizeXY, 0:boxSizeZ]  = intensite(d_volume[xMin : xMax, y, zMin : zMax])
            planYZ[0:boxSizeXY, 0:boxSizeZ]  = intensite(d_volume[x , yMin : yMax, zMin : zMax ])
        else:
            planXY[0:boxSizeXY, 0:boxSizeXY]  = d_volume[xMin : xMax, yMin : yMax, z ]
            planXZ[0:boxSizeXY, 0:boxSizeZ]  = d_volume[xMin : xMax, y, zMin : zMax]
            planYZ[0:boxSizeXY, 0:boxSizeZ]  = d_volume[x , yMin : yMax, zMin : zMax ]

    min = planXY.min()
    max = planXY.max()
    planXY = (planXY - min) * 255 / (max - min)
            
    min = planXZ.min()
    max = planXZ.max()
    planXZ = (planXZ - min) * 255 / (max - min)

    min = planYZ.min()
    max = planYZ.max()
    planYZ = (planYZ - min) * 255 / (max - min)

    planYZ.reshape((boxSizeXY, boxSizeZ))

    return np.concatenate((planXY, planXZ, planYZ), axis = 1)


def calc_holo_moyen(dirPath, sizeX, sizeY, extension):

    os.chdir(dirPath)
    holo_m = np.empty((sizeY,sizeX), dtype = np.float32)
    nb_images_tot = len(os.listdir(dirPath))
    nb_images = 0
    start = time.time()

    for image in os.listdir(dirPath):
        if (image.split('.')[-1].lower() == extension.lower()):
            nb_images +=1
            img= Image.open(image)
            holo = np.asarray(img)

            sx = np.size(holo, axis=1)
            sy = np.size(holo, axis=0)

            offsetX = (sx - sizeX)//2
            offsetY = (sy - sizeY)//2
            
            holo = holo[offsetY:offsetY+sizeY:1, offsetX:offsetX+sizeX:1]
            holo_m += holo
            img.close()
            print(round(100* nb_images/nb_images_tot, 1), "% Done")

    print("Nombre d'images: ", nb_images)
    print("Temps d'execution", time.time() - start)

    holo_m = holo_m / nb_images

    return(holo_m)


def analyse_array_cplx(data):
    if isinstance(data, cp.ndarray):
        h_data = intensite(cp.asnumpy(data))
    else:
        h_data = intensite(data)
    
    min = h_data.min()
    max = h_data.max()
    sum = h_data.sum()
    mean = h_data.mean()
    std = h_data.std()
    print('min = ', min, 'max = ', max, 'sum =', sum, 'mean = ', mean, 'std =', std)
    return(min, max, mean, sum, std)

def analyse_array(data, titre = ""):
    if isinstance(data, cp.ndarray):
        h_data = cp.asnumpy(data)
    else:
        h_data = data
    
    min = h_data.min()
    max = h_data.max()
    sum = h_data.sum()
    mean = h_data.mean()
    std = h_data.std()
    print(titre, ' min = ', min, 'max = ', max, 'sum =', sum, 'mean = ', mean, 'std =', std)
    return(min, max, mean, sum, std)

def sum_plans(d_volum_focus):
    return(d_volum_focus.sum(axis = 0), d_volum_focus.sum(axis = 1), d_volum_focus.sum(axis = 2))


@jit.rawkernel()
def d_filter_FFT_3D(d_VOLUME_IN, d_VOLUME_OUT, sizeX, sizeY, sizeZ, dMinXY, dMaxXY, dMinZ, dMaxZ):

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    planSize = sizeX * sizeY
    kk = index // planSize
    jj = ( index - kk * planSize )// sizeX
    ii = index - jj * sizeX - kk * planSize

    if (ii < sizeX and jj < sizeY):
        #calc distance
        centreX = sizeX // 2
        centreY = sizeY // 2
        centreZ = sizeZ // 2

        distanceCentre = cp.sqrt((centreX - ii)*(centreX - ii) + (centreY - jj)*(centreY - jj))
        distanceZ = cp.abs(centreZ - kk)

        if ((distanceCentre > dMinXY) and (distanceCentre < dMaxXY ) and (distanceZ > dMinZ) and (distanceZ < dMaxZ )):
            d_VOLUME_OUT[ii, jj, kk] = d_VOLUME_IN[ii, jj, kk]
        else:
            d_VOLUME_OUT[ii, jj, kk] = 0.0 + 0.0j

def filtre_volume(d_FFT_volume_IN, d_FFT_volume_OUT, sizeX, sizeY, sizeZ, dMinXY, dMaxXY, dMinZ, dMaxZ):

    nthread = 1024
    nBlock = math.ceil(sizeX * sizeX * sizeZ // nthread)

    d_filter_FFT_3D[nBlock, nthread](d_FFT_volume_IN, d_FFT_volume_OUT, sizeX, sizeY, sizeZ, dMinXY, dMaxXY, dMinZ, dMaxZ)

def normalise_to_U8_volume(d_volume_IN):

    min = cp.min(d_volume_IN)
    max = cp.max(d_volume_IN)

    #d_volume_out = cp.zeros(dtype = cp.uint8, shape = d_volume_IN.shape)

    return(((d_volume_IN - min) * 255 / (max - min)).astype(cp.uint8))


def projection_bool(d_bin_volume, axis):

    sizeX, sizeY, sizeZ = d_bin_volume.shape
    if axis == 0:
        d_projection = cp.zeros(shape=(sizeY, sizeZ), dtype=cp.uint8)
        nthread = 1024
        nBlock = math.ceil(sizeY * sizeZ // nthread)
    elif axis == 1:
        d_projection = cp.zeros(shape=(sizeX, sizeZ), dtype=cp.uint8)
        nthread = 1024
        nBlock = math.ceil(sizeX * sizeZ // nthread)
    elif axis == 2:
        d_projection = cp.zeros(shape=(sizeX, sizeY), dtype=cp.uint8)
        nthread = 1024
        nBlock = math.ceil(sizeX * sizeY // nthread)

    d_projection_bool[nBlock, nthread](d_bin_volume, d_projection, sizeX, sizeY, sizeZ, axis)

    return d_projection

@jit.rawkernel()
def d_projection_bool(d_bin_volume, d_projection, sizeX, sizeY, sizeZ, axis):

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    if axis == 0:
        jj = int(index // sizeZ)
        kk = int(index - jj * sizeZ)
        if (jj < sizeY and kk < sizeZ):
            val = 0
            for ii in range(sizeX):
                if d_bin_volume[ii, jj, kk]:
                    val = 1
            d_projection[jj, kk] = val

    elif axis == 1:
        ii = int(index // sizeZ)
        kk = int(index - ii * sizeZ)
        if (ii < sizeX and kk < sizeZ):
            val = 0
            for jj in range(sizeY):
                if d_bin_volume[ii, jj, kk]:
                    val = 1
            d_projection[ii, kk] = val

    elif axis == 2:
        ii = int(index // sizeY)
        jj = int(index - ii * sizeY)
        if (ii < sizeX and jj < sizeY):
            val = 0
            for kk in range(sizeZ):
                if d_bin_volume[ii, jj, kk]:
                    val = 1
            d_projection[ii, jj] = val





    

