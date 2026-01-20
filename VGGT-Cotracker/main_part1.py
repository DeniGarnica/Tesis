import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("name", nargs="?", default="axolot")
args = parser.parse_args()

scripts = [
    (None,     "VGGT-Cotracker/1-vggt_outputs_run.py"),
    (None,     "VGGT-Cotracker/2-Cotracker_run.py"),
    ("Samv2",  "VGGT-Cotracker/3-sam_masks_run.py"),
    (None,     "VGGT-Cotracker/4-use_mask.py"),
    (None,     "VGGT-Cotracker/5-TrackTo3D.py"),
    (None,     "VGGT-Cotracker/6-to3D.py"),
]

n_steps = len(scripts)

for i, (env, script) in enumerate(scripts, start=1):
    if env is None:
        print(f"[Paso {i}/{n_steps}] Ejecutando {script} (entorno actual)")
        cmd = [sys.executable, script, args.name]
    else:
        print(f"[Paso {i}/{n_steps}] Ejecutando {script} (conda env: {env})")
        cmd = ["conda", "run", "-n", env, "python", script, args.name]

    subprocess.run(cmd, check=True)
