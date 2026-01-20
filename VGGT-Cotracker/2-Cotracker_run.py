import torch
import imageio.v3 as iio
import glob
import os
import argparse

# Paths
parser = argparse.ArgumentParser()
parser.add_argument("name", nargs="?", default="axolot")

args = parser.parse_args()
name = args.name

frames_dir = f"data/{name}"

# Filtrar solo imágenes (puedes ajustar extensión a .jpg o .jpeg si es el caso)
frames_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))

print(f"Se encontraron {len(frames_paths)} frames")

# Cargar todos los frames
frames_list = [iio.imread(path) for path in frames_paths]

# Convertir a tensor: (T, H, W, C)
frames = torch.tensor(frames_list)

device = "cuda"
grid_size = 20

# Reordenar a (B, T, C, H, W)
video = frames.permute(0, 3, 1, 2)[None].float().to(device)

# Cargar CoTracker 3
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)

# Ejecutar tracking
pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size)

import numpy as np

# Extraer los tracks visibles en TODOS los frames
# pred_tracks: (B, T, N, 2) → asumimos batch=1
tracks_all = pred_tracks[0].detach().cpu().numpy()   # (T, N, 2)

vis_all = pred_visibility[0].detach().cpu().numpy().astype(float)

# Máscara: que se ven en TODOS los frames
mask_points = (vis_all > 0.5).all(axis=0)  

# Aplicar máscara para quedarnos solo con los puntos válidos
tracks_masked = tracks_all[:, mask_points, :]        # (T, N_valid, 2)

print("Forma final de tracks guardados:", tracks_masked.shape)
# Ejemplo: (T=3, N_valid=50, 2)

# Guardar en .npy (binario, conserva floats con precisión)
os.makedirs(f"outputs/{name}", exist_ok=True)
np.save(f"outputs/{name}/{name}_Cotracked_points.npy", tracks_masked)
