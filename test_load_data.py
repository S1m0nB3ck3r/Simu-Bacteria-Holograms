import simu_hologram as simu_holo
import traitement_holo as traitement_holo
import numpy as np
import os


chemin = r"C:\TRAVAIL\RepositoriesGithub\Simu-Bacteria-Holograms\simu_bact_random\2025_06_27_10_30_04\test\holo_140.npz"

bool_volume, hologram_image, parameters, bacteria_list= simu_holo.load_holo_data(chemin)

traitement_holo.affichage(bool_volume.sum(axis=2))
traitement_holo.affichage(hologram_image)

print("Parameters:", parameters)

# Parameters = {
#     'holo_size_x': 1024,
#     'holo_size_y': 1024,
#     'border_szie': 256,
#     'upscale_factor': 2, 
#     'holo_plane_number': 100, 
#     'medium_index': 1.33, 
#     'object_index': 1.335, 
#     'pix_size_cam': 5.5e-06, 
#     'magnification_cam': 40, 
#     'Z_step': 1e-06, 
#     'illumination_wavelength': 6.6e-07, 
#     'medium_wavelength': 4.962406015037594e-07}