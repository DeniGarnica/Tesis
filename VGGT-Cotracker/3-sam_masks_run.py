import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("name", nargs="?", default="axolot")

args = parser.parse_args()
name = args.name

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "VGGT-Cotracker/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

import glob

frames_dir = f"data/{name}"
os.makedirs(f"outputs/{name}", exist_ok=True)
Cootracked_path = f"outputs/{name}/{name}_Cotracked_points.npy"

# Filtrar solo imágenes (puedes ajustar extensión a .jpg o .jpeg si es el caso)
frames_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
cotracked_points = np.load(Cootracked_path)    
print(f"Se encontraron {cotracked_points.shape[1]} puntos cotracked en {cotracked_points.shape[0]} frames")
print(f"Se encontraron {len(frames_paths)} frames")


import os
import cv2

def process_points(points, image, output_dir, id_frame, base_name="mask", min_area=0.15):
    """
    Procesa una lista de puntos en una imagen, genera máscaras binarias,
    evita recalcular si el punto cae en una máscara ya procesada,
    descarta máscaras demasiado pequeñas
    y fuerza a que cada máscara sea conexa (componente que contiene el punto).

    Parámetros:
    -----------
    points : list of (x, y)
        Coordenadas de los puntos en la imagen.
    image : np.ndarray
        Imagen sobre la cual se procesan los puntos.
    output_dir : str
        Carpeta donde se guardarán las máscaras.
    base_name : str
        Nombre base para los archivos guardados.
    min_area : float
        Proporción mínima del área total de la imagen para aceptar la máscara.
    """

    os.makedirs(output_dir, exist_ok=True)
    processed_masks = []   # lista de máscaras ya generadas
    predictor.set_image(image)

    H, W = image.shape[:2]
    min_pixels = min_area * H * W

    for idx, pt in enumerate(points):
        x, y = map(int, pt)

        # Revisar si el punto ya pertenece a alguna máscara existente
        skip = False
        for m in processed_masks:
            if m[y, x] == 1:   # está dentro de una máscara ya procesada
                skip = True
                break

        if skip:
            continue

        # Generar nueva máscara
        masks, scores, logits = predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        mask = masks[0].astype(np.uint8)

        # --- Filtrar componente conexa que contiene al punto ---
        num_labels, labels = cv2.connectedComponents(mask)
        if num_labels <= 1:  # no hay componentes
            continue

        comp_id = labels[y, x]  # id de la componente donde cae el punto
        if comp_id == 0:  # el punto cayó en fondo
            continue

        # Crear nueva máscara solo con esa componente
        mask = (labels == comp_id).astype(np.uint8)

        # Filtrado por área mínima
        area = mask.sum()
        if area < min_pixels:
            continue

        # Guardar como .npy
        out_npy = os.path.join(output_dir, f"{base_name}_f{id_frame}_{len(processed_masks):03d}.npy")
        np.save(out_npy, mask)

        # Guardar como imagen PNG (0/255)
        mask_img = mask * 255
        out_png = os.path.join(output_dir, f"{base_name}_f{id_frame}_{len(processed_masks):03d}.png")
        cv2.imwrite(out_png, mask_img)

        # Agregar la máscara válida a la lista
        processed_masks.append(mask)

    print(f"Procesados {len(processed_masks)} máscaras únicas (≥{min_area*100:.2f}% área).")



output_dir = f"masks_all/{name}"

for i, (frame_path, tracked_points) in enumerate(zip(frames_paths, cotracked_points)):

    print(f"Procesando {frame_path}...")
    
    image = Image.open(frame_path)
    image = np.array(image.convert("RGB"))

    min_area = .05  # Proporción mínima del área total de la imagen

    process_points(tracked_points, image, output_dir, id_frame=i, base_name="mask", min_area=min_area)