import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os
import cv2
import numpy as np
import open3d as o3d
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import argparse

# Paths
parser = argparse.ArgumentParser()
parser.add_argument("name", nargs="?", default="axolot")

args = parser.parse_args()
name = args.name

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

model = model.to(device)

os.makedirs(f"outputs/{name}", exist_ok=True)
image_names = [f"data/{name}/000000.png", 
               f"data/{name}/000001.png", 
               f"data/{name}/000002.png", 
               f"data/{name}/000003.png", 
               f"data/{name}/000004.png", 
               f"data/{name}/000005.png"]  
images = load_and_preprocess_images(image_names).to(device)
print(f"Preprocessed images shape: {images.shape}")

# Función para guardar nube de puntos con colores RGB
def save_ply(points, colors, filename):
    if torch.is_tensor(points):
        points_visual = points.reshape(-1, 3).cpu().numpy()
    else:
        points_visual = points.reshape(-1, 3)
    if torch.is_tensor(colors):
        points_visual_rgb = colors.reshape(-1, 3).cpu().numpy()
    else:
        points_visual_rgb = colors.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_visual.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(points_visual_rgb.astype(np.float64))
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)
                
    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    # Predict Point Maps
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
        
    # Construct 3D Points from Depth Maps and Cameras
    # which usually leads to more accurate 3D points than point map branch
    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                extrinsic.squeeze(0), 
                                                                intrinsic.squeeze(0))
    
    # Guardar nubes de puntos con color RGB original por cada imagen
    os.makedirs("outputs", exist_ok=True)
    for i in range(point_map_by_unprojection.shape[0]):
        save_ply(
            point_map_by_unprojection[i],            # (H, W, 3) puntos
            images[0, i].permute(1, 2, 0),           # (H, W, 3) imagen RGB
            f"outputs/{name}/{name}_cloud_{i}.ply"
        )

    # Generamos puntos para seguimiento (rejilla regular)
    H, W = int(images.shape[-2]), int(images.shape[-1])
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H),  # 0 a H-1
        torch.arange(W),  # 0 a W-1
        indexing="ij"
    )

    query_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1).to(device)
    # Parámetros para dividir en batches
    query_points = query_points[None]  # (1, N, 2)
    batch_size = 2000  # Ajustar este valor según la memoria GPU
    num_points = query_points.shape[1]
    num_batches = (num_points + batch_size - 1) // batch_size

    # Inicializamos listas para guardar resultados
    tracked_batches = []
    vis_batches = []
    conf_batches = []

    # Seguimiento por lotes (batches)
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_points)
        query_batch = query_points[:, start:end, :]

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                tracked, vis, conf = model.track_head(
                    aggregated_tokens_list, images, ps_idx,
                    query_points=query_batch
                )
        print(f"Porcentaje de seguimiento completado: {100 * (i + 1) / num_batches:.2f}%", end="\r")
        tracked_batches.append(tracked[0].squeeze(0).cpu().float())
        vis_batches.append(vis[0].squeeze(0).cpu().float())
        conf_batches.append(conf[0].squeeze(0).cpu().float())


    # Concatenamos en la dimensión de los puntos (dim=1)
    tracked_points = torch.cat(tracked_batches, dim=1).numpy()  # (2, N, 2)
    vis = torch.cat(vis_batches, dim=1).numpy()                # (2, N)
    conf = torch.cat(conf_batches, dim=1).numpy()              # (2, N)

    np.save(f"outputs/{name}/{name}_tracked_points.npy", tracked_points)    
    np.save(f"outputs/{name}/{name}_visibility.npy", vis)
    np.save(f"outputs/{name}/{name}_confidence.npy", conf)

# Guardamos las matrices intrínsecas y extrínsecas
np.save(f"outputs/{name}/{name}_extrinsic.npy", extrinsic.squeeze().cpu().numpy())
np.save(f"outputs/{name}/{name}_intrinsic.npy", intrinsic.squeeze().cpu().numpy())
np.save(f"outputs/{name}/{name}_depth_map.npy", depth_map.cpu().numpy())
np.save(f"outputs/{name}/{name}_depth_conf.npy", depth_conf.cpu().numpy())
np.save(f"outputs/{name}/{name}_point_map_by_unprojection.npy", point_map_by_unprojection)

print("Predicciones disponibles:", predictions.keys())