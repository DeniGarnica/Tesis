import os
import csv
import json
import numpy as np
import open3d as o3d
import colorsys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("name", nargs="?", default="axolot")

args = parser.parse_args()
name = args.name

# -------- Colores HSV determinísticos por Punto_ID --------
GOLDEN_RATIO_CONJ = 0.618033988749895

def rgb_from_point_id(pid: int, s: float = 0.9, v: float = 0.95):
    """
    Color estable por Punto_ID:
    h = (pid * phi_conj) % 1  -> HSV -> RGB en [0,1]
    """
    h = (pid * GOLDEN_RATIO_CONJ) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (float(r), float(g), float(b))


def cargar_centroides(csv_path, archivos_filtrados=None):
    """
    Lee puntos 3D del CSV:
    columnas: Archivo_cloud, Punto_ID, X, Y, Z.
    Devuelve lista: (frame_int, punto_id, np.array([x,y,z])) sólo de los archivos deseados.
    """
    puntos = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            archivo = row["Archivo_cloud"]  # ej. "000000_cloud.ply"
            if archivos_filtrados is not None and archivo not in archivos_filtrados:
                continue
            # frame: convierte "name_cloud_0.ply" -> 0
            frame = int(archivo.split("_")[2][0])
            punto_id = int(row["Punto_ID"])
            x, y, z = float(row["X"]), float(row["Y"]), float(row["Z"])
            puntos.append((frame, punto_id, np.array([x, y, z], dtype=float)))

    puntos.sort(key=lambda x: (x[0], x[1]))  # orden por frame y luego por punto_id
    print(f"CSV: {csv_path} | puntos cargados: {len(puntos)} | archivos: {archivos_filtrados}")
    return puntos


def crear_visualizacion(puntos, out_path):
    """
    Genera:
      - *_points.ply con colores por Punto_ID
      - *_lines.ply con trayectorias coloreadas por Punto_ID
    Conserva colores HSV coherentes para el mismo Punto_ID.
    """
    if len(puntos) == 0:
        print("No hay puntos para visualizar.")
        return

    coords = [p[2] for p in puntos]

    # Traza índices por Punto_ID a lo largo de frames
    trayectorias = {}
    for idx, (frame, punto_id, coord) in enumerate(puntos):
        trayectorias.setdefault(punto_id, []).append((frame, idx))

    # Mapa estable Punto_ID -> color RGB (en [0,1])
    unique_ids = sorted(trayectorias.keys())
    id_to_color = {pid: rgb_from_point_id(pid) for pid in unique_ids}

    # Colores por vértice (según Punto_ID)
    vertex_colors = [id_to_color[p[1]] for p in puntos]

    # Líneas entre observaciones consecutivas del mismo Punto_ID
    line_indices, line_colors = [], []
    for punto_id, traj in trayectorias.items():
        traj.sort(key=lambda x: x[0])  # por frame
        for i in range(len(traj) - 1):
            idx1 = traj[i][1]
            idx2 = traj[i+1][1]
            line_indices.append([idx1, idx2])
            line_colors.append(id_to_color[punto_id])

    # Ensambles Open3D
    cloud = o3d.geometry.PointCloud()
    cloud.points  = o3d.utility.Vector3dVector(np.asarray(coords))
    cloud.colors  = o3d.utility.Vector3dVector(np.asarray(vertex_colors))

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(coords)),
        lines=o3d.utility.Vector2iVector(np.asarray(line_indices, dtype=np.int32)) if line_indices else o3d.utility.Vector2iVector()
    )
    if line_colors:
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(line_colors))

    # Guardar
    out_points = out_path.replace(".ply", "_points.ply")
    out_lines  = out_path.replace(".ply", "_lines.ply")
    o3d.io.write_point_cloud(out_points, cloud)
    o3d.io.write_line_set(out_lines, line_set)

    print(f"Guardado:\n- {out_points}\n- {out_lines}")


if __name__ == "__main__":

    csv_path  = f"outputs/{name}/cotracked_points_3d_world.csv"
    out_path  = f"outputs/{name}/{name}_tracks_with_points.ply"
    # (Opcional, no usado aquí) depth_map_path = f"{base_path}/{name}/outputs/{name}_depth_map.npy"

    # Solo usar estas nubes:
    archivos_usar = [
        f"{name}_cloud_0.ply", 
        f"{name}_cloud_1.ply", 
        f"{name}_cloud_2.ply",
        f"{name}_cloud_3.ply", 
        f"{name}_cloud_4.ply",
        f"{name}_cloud_5.ply"
        ]

    puntos = cargar_centroides(csv_path, archivos_filtrados=archivos_usar)
    crear_visualizacion(puntos, out_path)
