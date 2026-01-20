import numpy as np
import imageio
import argparse

# Paths
parser = argparse.ArgumentParser()
parser.add_argument("name", nargs="?", default="axolot")

args = parser.parse_args()
name = args.name
points_path = f"outputs/{name}/{name}_Cotracked_points.npy"
masks = [
        f"masks_all/{name}/mask_f0_001.png",
        f"masks_all/{name}/mask_f1_002.png",
        f"masks_all/{name}/mask_f2_002.png",
        f"masks_all/{name}/mask_f3_001.png",
        f"masks_all/{name}/mask_f4_002.png",
        f"masks_all/{name}/mask_f5_001.png"
]

# Cargar puntos (T, N, 2)
points = np.load(points_path)  # (6, 53, 2)
T, N, _ = points.shape
print(f"Puntos cargados: {points.shape}")

# Cargar máscaras binarias (T, H, W)
masks_bin = [imageio.imread(m).astype(bool) for m in masks]

# Filtrar puntos
filtered_points = []
for t in range(T):
    mask = masks_bin[t]
    pts_frame = []
    for i in range(N):
        u, v = points[t, i]
        u, v = int(round(u)), int(round(v))
        # Validar que esté dentro de la máscara y que sea True
        if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]:
            if mask[v, u]:
                pts_frame.append([u, v])
    filtered_points.append(np.array(pts_frame))

# Guardar como objeto (cada frame puede tener distinto número de puntos)
filtered_points = np.stack(filtered_points, axis=0)  # shape (T, N, 2)
out_path = points_path.replace(".npy", "_filtered.npy")
np.save(out_path, filtered_points)

print(f"Guardado en: {out_path}")
for t, pts in enumerate(filtered_points):
    print(f"Frame {t}: {len(pts)} puntos")
