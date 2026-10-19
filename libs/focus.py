# -*- coding: utf-8 -*-

"""
Filename: focus.py

Description:
Groups of functions needed for treat 3d holograms with a focus criterion to make appear objects of interest.
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
from cupyx.scipy import ndimage as cp_ndimage
from . traitement_holo import *

from enum import Enum
class Focus_type(Enum):
    ALL = 1
    SUM_OF_LAPLACIAN = 2
    SUM_OF_VARIANCE = 3
    TENEGRAD = 4
    SUM_OF_INTENSITY = 5
    SUM_OF_GRADIENT = 6
    MEAN_ALL = 7
    MEAN_LOG_ALL = 8
    VAR_IMAG_OVER_REAL = 9     # exige un volume COMPLEXE (voir focus_var_imag_over_real)

# critères qui exigent le champ complexe reconstruit, et non son module
COMPLEX_FOCUS_TYPES = (Focus_type.VAR_IMAG_OVER_REAL,)

def focus_sum_of_gradient(d_volume_IN, d_focus_OUT, sumSize):
    """
    Compute sum of squared gradient (|dx|^2 + |dy|^2) per plane and local average.
    d_volume_IN : 3D array (Z,Y,X) (cupy)
    d_focus_OUT  : 3D array (Z,Y,X) output (cupy)
    sumSize      : int, window size for local averaging (will be made odd)
    """
    sizeZ, sizeY, sizeX = cp.shape(d_volume_IN)

    # allocate temporary planes
    module_plane = cp.zeros(shape=(sizeY, sizeX), dtype=cp.float32)
    gradx_plane = cp.zeros(shape=(sizeY, sizeX), dtype=cp.float32)
    grady_plane = cp.zeros(shape=(sizeY, sizeX), dtype=cp.float32)

    # ensure odd sumSize
    sumSize = (sumSize // 2) * 2 + 1
    convolve_plane = cp.full(fill_value=1.0 / (sumSize * sumSize), dtype=cp.float32, shape=(sumSize, sumSize))

    if d_volume_IN.dtype == cp.complex64:
        for p in range(sizeZ):
            module_plane = module(d_volume_IN[p, :, :])
            # sobel returns gradient approximation
            gradx_plane = cp_ndimage.sobel(module_plane, axis=1)
            grady_plane = cp_ndimage.sobel(module_plane, axis=0)
            mag_plane = cp.square(gradx_plane) + cp.square(grady_plane)
            cp_ndimage.convolve(mag_plane, convolve_plane, output=d_focus_OUT[p, :, :], mode='reflect')
    else:
        for p in range(sizeZ):
            module_plane = d_volume_IN[p, :, :]
            gradx_plane = cp_ndimage.sobel(module_plane, axis=1)
            grady_plane = cp_ndimage.sobel(module_plane, axis=0)
            mag_plane = cp.square(gradx_plane) + cp.square(grady_plane)
            cp_ndimage.convolve(mag_plane, convolve_plane, output=d_focus_OUT[p, :, :], mode='reflect')

def focus_sum_square_of_laplacien(d_volume_IN, d_focus_OUT, sumSize):
    
    sizeZ, sizeY, sizeX = cp.shape(d_volume_IN)

    #allocation intensity plane
    module_plane = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)
    laplace_plane = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)

    #si sumSize est pair rajouter 1
    sumSize = (sumSize //2) * 2  + 1
    #allocation plan de convolution (carré de 1.0 de taille sumSize)
    convolve_plane = cp.full(fill_value= 1.0 / (sumSize*sumSize), dtype=cp.float32, shape=(sumSize, sumSize))

    if d_volume_IN.dtype == cp.complex64:
        for p in range(sizeZ):
            module_plane = module(d_volume_IN[p,:,:])
            laplace_plane = cp_ndimage.laplace(module_plane)
            laplace_plane = cp.square(laplace_plane)
            # laplace_plane = laplace_plane / module_plane
            cp_ndimage.convolve(laplace_plane, convolve_plane, output = d_focus_OUT[p,:,:], mode = 'reflect')

    else : # module float32
        for p in range(sizeZ):
            module_plane = d_volume_IN[p,:,:]
            laplace_plane = cp_ndimage.laplace(module_plane)
            laplace_plane = cp.square(laplace_plane)
            # laplace_plane = laplace_plane / module_plane
            cp_ndimage.convolve(laplace_plane, convolve_plane, output = d_focus_OUT[p,:,:], mode = 'reflect')
        

def focus_sum_of_variance(d_volume_IN, d_focus_OUT, sumSize):
    sizeZ, sizeY, sizeX = cp.shape(d_volume_IN)

    ### calcul du focus selon la méthode sumOfVariance
    ### REF Autofocusing in digital holographic phase contrast microscopy on pure phase objects for live cell imaging
    ### dans tout le volume, pour chaque voxel est calcule la variance locale dans un plan XY (de taille sumSize) autour de celui-ci
    ### pour chaque plan:
    ### 1. calcul du plan d'amplitude (qui marche mieux que plan d'intensité)
    ### 2. calcul d'un plan de moyenne locale de l'amplitude (taille sumSize) par une convolution d'un kernel de valeurs 1/sumSize
    ### 3. calcul d'un plan d'écart à la moyenne locale au carré
    ### 4. convolution du plan précédent ->kernel de taille sumSize de valeur 1/sumSize

    #allocation intensity plane
    module_plane = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)
    local_mean_plane = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)
    variance_plane = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)

    #si sumSize est pair rajouter 1
    sumSize = (sumSize //2 ) * 2 + 1
    #allocation plan de convolution (carré de 1.0 de taille sumSize)
    convolve_mean = cp.full(fill_value=1.0 / (sumSize*sumSize), dtype=cp.float32, shape=(sumSize, sumSize))

    for p in range(sizeZ):
        module_plane = module(d_volume_IN[p,:,:])
        cp_ndimage.convolve(module_plane, convolve_mean, output = local_mean_plane, mode = 'reflect')
        # variance_plane = cp.square( local_mean_plane - module_plane ) / local_mean_plane

        variance_plane = cp.square( local_mean_plane - module_plane )
        cp_ndimage.convolve(variance_plane, convolve_mean, output = d_focus_OUT[p,:,:], mode = 'reflect')
        

def focus_TENEGRAD(d_volume_IN, d_focus_OUT, sumSize):
    sizeZ, sizeY, sizeX = cp.shape(d_volume_IN)

    #allocation intensity plane
    plane_module = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)
    plane_ten1 = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)
    plane_ten2 = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)

    ten1 = cp.asarray(((-1.0, 0.0, 1.0),(-2.0, 0.0, 2.0),(-1.0, 0.0, 1.0)), dtype = np.float32)
    ten2 = cp.asarray(((-1.0, -2.0, -1.0),(0.0, 0.0, 0.0),(1.0, 2.0, 1.0)), dtype = np.float32)

    #si sumSize est pair rajouter 1
    sumSize = (sumSize //2 )*2 +1
    #allocation plan de convolution (carré de 1.0 de taille sumSize)
    convolve_plane = cp.full(fill_value=1.0 / (sumSize * sumSize), dtype=cp.float32, shape=(sumSize, sumSize))

    if d_volume_IN.dtype == cp.complex64:
        for p in range(sizeZ):
            plane_module = module(d_volume_IN[p,:,:])
            cp_ndimage.convolve(plane_module, ten1, output = plane_ten1, mode = 'reflect')
            cp_ndimage.convolve(plane_module, ten2, output = plane_ten2, mode = 'reflect')
            plane_tenegard = cp.sqrt(plane_ten1**2 + plane_ten2**2)
            # plane_tenegard = plane_ten1**2 + plane_ten2**2

            cp_ndimage.convolve(plane_tenegard, convolve_plane,  output = d_focus_OUT[p,:,:], mode = 'reflect')
    else : # module float32
        for p in range(sizeZ):
            cp_ndimage.convolve(d_volume_IN[p,:,:], ten1, output = plane_ten1, mode = 'reflect')
            cp_ndimage.convolve(d_volume_IN[p,:,:], ten2, output = plane_ten2, mode = 'reflect')
            plane_tenegard = cp.sqrt(plane_ten1**2 + plane_ten2**2)
            # plane_tenegard = plane_ten1**2 + plane_ten2**2
            cp_ndimage.convolve(plane_tenegard, convolve_plane,  output = d_focus_OUT[p,:,:], mode = 'reflect')

def focus_SUM_OF_INTENSITY(d_volume_IN, d_focus_OUT, sumSize):

    sizeZ, sizeY, sizeX  = cp.shape(d_volume_IN)

    #allocation intensity plane
    plane = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)

    #si sumSize est pair rajouter 1
    sumSize = (sumSize //2 ) * 2 + 1
    #allocation plan de convolution (carré de 1.0 de taille sumSize)
    convolve_plane = cp.full(fill_value=1.0 / (sumSize * sumSize), dtype=cp.float32, shape=(sumSize, sumSize))

    if d_volume_IN.dtype == cp.complex64:
        for p in range(sizeZ):
            plane = intensite(d_volume_IN[p,:,:])
            cp_ndimage.convolve(plane, convolve_plane, output = d_focus_OUT[p,:,:], mode = 'reflect')
    else : # module float32
        for p in range(sizeZ):
            plane = d_volume_IN[p,:,:]**2
            cp_ndimage.convolve(plane, convolve_plane, output = d_focus_OUT[p,:,:], mode = 'reflect')

def focus_MEAN_ALL(d_volume_IN, d_focus_OUT, sumSize, d_accumulator=None):
    """
    Calcule la moyenne de tous les types de focus
    Traitement plan par plan pour économiser la mémoire GPU
    """
    nb_planes, height, width = d_volume_IN.shape
    
    # Allocate 2D temporary buffers (one plane only) - économise beaucoup de mémoire
    temp_plane = cp.zeros((height, width), dtype=cp.float32)
    accumulator_plane = cp.zeros((height, width), dtype=cp.float32)
    
    # Process each plane independently
    for plane_idx in range(nb_planes):
        accumulator_plane[:] = 0  # Reset accumulator for this plane
        count = 0
        
        # Apply each focus method to this single plane
        for focus_method in [focus_SUM_OF_INTENSITY, focus_sum_square_of_laplacien, 
                             focus_sum_of_variance, focus_TENEGRAD, focus_sum_of_gradient]:
            # Create a view of the single plane as a pseudo-3D volume (1, H, W)
            plane_view_in = d_volume_IN[plane_idx:plane_idx+1, :, :]
            temp_plane_view = temp_plane[cp.newaxis, :, :]  # Add dimension for compatibility
            
            focus_method(plane_view_in, temp_plane_view, sumSize)
            
            # Accumulate result (remove the extra dimension)
            accumulator_plane[:] = accumulator_plane + temp_plane_view[0, :, :]
            count += 1
        
        # Store averaged result for this plane
        d_focus_OUT[plane_idx, :, :] = accumulator_plane / count

def focus_MEAN_LOG_ALL(d_volume_IN, d_focus_OUT, sumSize, d_accumulator=None):
    """
    Calcule la moyenne logarithmique de tous les types de focus
    moyenne_log = exp(mean(log(x))) pour éviter overflow/underflow
    Traitement plan par plan pour économiser la mémoire GPU
    """
    nb_planes, height, width = d_volume_IN.shape
    
    # Allocate 2D temporary buffers (one plane only) - économise beaucoup de mémoire
    temp_plane = cp.zeros((height, width), dtype=cp.float32)
    accumulator_plane = cp.zeros((height, width), dtype=cp.float32)
    
    # Process each plane independently
    for plane_idx in range(nb_planes):
        accumulator_plane[:] = 0  # Reset accumulator for this plane
        count = 0
        
        # Apply each focus method to this single plane
        for focus_method in [focus_SUM_OF_INTENSITY, focus_sum_square_of_laplacien, 
                             focus_sum_of_variance, focus_TENEGRAD, focus_sum_of_gradient]:
            # Create a view of the single plane as a pseudo-3D volume (1, H, W)
            plane_view_in = d_volume_IN[plane_idx:plane_idx+1, :, :]
            temp_plane_view = temp_plane[cp.newaxis, :, :]  # Add dimension for compatibility
            
            focus_method(plane_view_in, temp_plane_view, sumSize)
            
            # Accumulate log of result (remove the extra dimension)
            # Ajouter un epsilon pour éviter log(0)
            accumulator_plane[:] = accumulator_plane + cp.log(temp_plane_view[0, :, :] + 1e-10)
            count += 1
        
        # Store averaged result for this plane (exp of mean of logs)
        d_focus_OUT[plane_idx, :, :] = cp.exp(accumulator_plane / count)

def focus_plane_var_imag_over_real(plane_complex, sumSize):
    """
    Critère de focus pour objet de phase, sur UN plan complexe :
        var_locale(Im E) / var_locale(Re E)
    variances locales sur une fenêtre carrée sumSize × sumSize (E[x²] − E[x]²).

    Au plan de focus d'un objet de phase (t = exp(iφ) ≈ 1 + iφ), la perturbation est
    imaginaire : la variance de Im domine. Hors focus, la propagation convertit la phase en
    amplitude, Re et Im se mélangent et le rapport tend vers 1, comme dans le bruit. Un objet
    absorbant donne au contraire un rapport < 1 au focus.

    Le plan doit être référencé en phase : porteuse exp(2iπ z/λ) retirée (voir
    propagation.volume_propag_angular_spectrum_complex(remove_carrier=True)), sinon Re et Im
    tournent d'un plan à l'autre. Le rapport est sans unité et ne dépend pas de l'éclairage.
    """
    sumSize = (sumSize // 2) * 2 + 1
    re = cp.real(plane_complex).astype(cp.float32)
    im = cp.imag(plane_complex).astype(cp.float32)
    var_re = cp_ndimage.uniform_filter(re * re, size=sumSize, mode='reflect') -         cp.square(cp_ndimage.uniform_filter(re, size=sumSize, mode='reflect'))
    var_im = cp_ndimage.uniform_filter(im * im, size=sumSize, mode='reflect') -         cp.square(cp_ndimage.uniform_filter(im, size=sumSize, mode='reflect'))
    var_re = cp.maximum(var_re, 0)
    var_im = cp.maximum(var_im, 0)
    # plancher relatif : évite la division par une variance quasi nulle
    floor = cp.float32(1e-6) * cp.mean(var_re) + cp.float32(1e-30)
    return (var_im / cp.maximum(var_re, floor)).astype(cp.float32)


def focus_var_imag_over_real(d_volume_IN, d_focus_OUT, sumSize):
    """Version volume de focus_plane_var_imag_over_real : d_volume_IN complexe (Z, Y, X)."""
    if not cp.iscomplexobj(d_volume_IN):
        raise ValueError("VAR_IMAG_OVER_REAL exige un volume complexe (champ reconstruit, porteuse "
                         "retirée), pas le module : voir propagation.volume_propag_angular_spectrum_complex"
                         "(remove_carrier=True) ou volume_propag_angular_spectrum_to_focus_complex")
    for p in range(d_volume_IN.shape[0]):
        d_focus_OUT[p, :, :] = focus_plane_var_imag_over_real(d_volume_IN[p, :, :], sumSize)


def focus(d_volume_IN, d_focus_OUT, sumSize, type_of_focus):

    if type_of_focus == Focus_type.TENEGRAD:
        focus_TENEGRAD(d_volume_IN, d_focus_OUT, sumSize)
    elif type_of_focus == Focus_type.SUM_OF_VARIANCE:
        focus_sum_of_variance(d_volume_IN, d_focus_OUT, sumSize)
    elif type_of_focus == Focus_type.SUM_OF_LAPLACIAN:
        focus_sum_square_of_laplacien(d_volume_IN, d_focus_OUT, sumSize)
    elif type_of_focus == Focus_type.SUM_OF_INTENSITY:
        focus_SUM_OF_INTENSITY(d_volume_IN, d_focus_OUT, sumSize)
    elif type_of_focus == Focus_type.SUM_OF_GRADIENT:
        focus_sum_of_gradient(d_volume_IN, d_focus_OUT, sumSize)
    elif type_of_focus == Focus_type.MEAN_ALL:
        focus_MEAN_ALL(d_volume_IN, d_focus_OUT, sumSize)
    elif type_of_focus == Focus_type.MEAN_LOG_ALL:
        focus_MEAN_LOG_ALL(d_volume_IN, d_focus_OUT, sumSize)
    elif type_of_focus == Focus_type.VAR_IMAG_OVER_REAL:
        focus_var_imag_over_real(d_volume_IN, d_focus_OUT, sumSize)
