import os
import csv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("name", nargs="?", default="axolot")

args = parser.parse_args()
name = args.name

def pixel_to_3d(u, v, depth_map, K):
    """
    Convierte coordenadas de pixel (u, v) a un punto 3D usando el valor de profundidad
    y la matriz intrínseca K de la cámara.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Profundidad en la posición
    z = depth_map[int(v), int(u)]
    if z == 0:  # sin valor de profundidad válido
        return None

    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z
    return np.array([X, Y, Z])

def procesar_tracks(points_path, depth_path, K_path, E_path, out_dir, name, mask_shape=None):
    """
    Procesa un archivo .npy con puntos 2D trackeados en varios frames y
    convierte cada uno en un punto 3D (mundo).
    Usa el depth map y las intrínsecas/extrínsecas de la cámara.
    """
    # Cargar intrínsecos, extrínsecos y depth map
    K = np.load(K_path)          # (T, 3, 3)
    E = np.load(E_path)          # (T, 4, 4)
    depth_all = np.load(depth_path)  # (T, H, W)

    # Aseguramos que quede como (T, H, W)
    if depth_all.ndim == 5:  # (1, T, H, W, 1)
        depth_all = np.squeeze(depth_all, axis=(0, -1))
    elif depth_all.ndim == 4:  # (1, T, H, W)
        depth_all = np.squeeze(depth_all, axis=0)
    elif depth_all.ndim != 3:
        raise ValueError(f"Formato inesperado de depth map: {depth_all.shape}")

    print(f"Depth map cargado con forma: {depth_all.shape}")  # (T, H, W)
    H_depth, W_depth = depth_all.shape[1:3]

    # Cargar puntos 2D trackeados
    cotracked_points = np.load(points_path)  # (T, N, 2)
    print(f"Puntos trackeados: {cotracked_points.shape}")

    resultados = []
    T, N, _ = cotracked_points.shape
    for t in range(T):
        depth_map = depth_all[t]
        K_base = K[t]
        E_base = E[t]

        R = E_base[:3, :3]
        tvec = E_base[:3, 3]

        for i in range(N):
            u, v = cotracked_points[t, i]

            # Rescalar si las máscaras eran de otro tamaño
            if mask_shape is not None:
                H_mask, W_mask = mask_shape
                u *= (W_depth / W_mask)
                v *= (H_depth / H_mask)

            punto_cam = pixel_to_3d(u, v, depth_map, K_base)
            if punto_cam is None:
                continue

            # Cámara a Mundo
            punto_world = R.T @ (punto_cam - tvec)

            resultados.append((f"{name}_cloud_{t}.ply", i, *punto_world))
            print(f"Frame {t}, Punto {i}: cam={punto_cam} → world={punto_world}")

    # Guardar resultados
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cotracked_points_3d_world.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Archivo_cloud", "Punto_ID", "X", "Y", "Z"])
        writer.writerows(resultados)

    print(f"\nPuntos 3D (world) guardados en: {out_path}")

# MAIN
if __name__ == "__main__":

    points_path = f"outputs/{name}/{name}_Cotracked_points_filtered.npy"
    depth_path  = f"outputs/{name}/{name}_depth_map.npy"
    K_path      = f"outputs/{name}/{name}_intrinsic.npy"
    E_path      = f"outputs/{name}/{name}_extrinsic.npy"
    out_dir     = f"outputs/{name}" 

    # Detectamos tamaño de las máscaras leyendo una cualquiera
    masks_dir = f"masks_all/{name}"
    example_mask = None
    for f in os.listdir(masks_dir):
        if f.endswith(".npy"):
            example_mask = np.load(os.path.join(masks_dir, f))
            break
    if example_mask is None:
        raise RuntimeError("No se encontró ninguna máscara en la carpeta")

    H_mask, W_mask = example_mask.shape
    print(f"Dimensiones de máscara detectadas: {H_mask}x{W_mask}")

    procesar_tracks(points_path, depth_path, K_path, E_path, out_dir, name, mask_shape=(H_mask, W_mask))
