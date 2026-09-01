import os
import glob
import re
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import pyvips
from tqdm import tqdm

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading


SVS_FOLDER = [Path(r"/mnt/share/2025-09-24_OSR"), Path(r"/mnt/share/2025-09-29_OSR"), Path(r"/mnt/share/2025-09-30_OSR")]
PATCHES_ROOT = Path(r"/mnt/g/AIFlopp/Patch_Milano")

#CROP_COORDS_CSV = Path(r"/mnt/e/esecuzione/images_to_crop.csv")

#LEVEL = 0 -> se serve un livello diverso, scommentare gli altri LEVEL
PATCH_SIZE = 512
BACKGROUND_THRESHOLD = 0.70
WHITE_THRESHOLD = 220

PATCHES_ROOT.mkdir(parents=True, exist_ok=True)

# IDENTIFICAZIONE FILE
if isinstance(SVS_FOLDER, Path):
    svs_files = glob.glob(str(SVS_FOLDER / "**" / "*.svs"), recursive=True)
elif isinstance(SVS_FOLDER, list):
    svs_files = []
    for folder in SVS_FOLDER:
        svs_files.extend(glob.glob(str(folder / "**" / "*.svs"), recursive=True))

print(f"Trovati {len(svs_files)} file SVS\n")


# IDENTIFICAZIONE PAZIENTI
patients = defaultdict(lambda: defaultdict(dict))

for path in svs_files:
    filename = os.path.basename(path)

    # Regex universale per file con _ o - come separatore
    # Esempi validi:
    # RO-I-25-197-1-N-2_192341_xxxxn_JUP.svs
    # RE-I_25_16291_1_A_2_132717_prostate_HNE.svs
    #match = re.match(r"^([A-Z0-9_-]+)[-_]([A-Z])[-_](\d+)[_]", filename, re.IGNORECASE)

    # Supporta anche nomi come TN_1_1-113600.svs, dove il numero finale
    # puo' essere seguito direttamente dall'estensione .svs.
    match = re.match(r"^([A-Z0-9_-]+)[-_]([A-Z0-9]+)[-_](\d+)(?:[_\.]|$)", filename, re.IGNORECASE)

    if match:
        # FIXME: controllare questo
        patient_id = match.group(1).upper().replace("-", "_")  # normalizza trattini -> underscore
        letter = match.group(2).upper()
        level_num = int(match.group(3))

        patients[patient_id][letter][level_num] = path
    else:
        print(f"Nome non riconosciuto: {filename}")

print(f"Pazienti trovati (chiavi unificate): {len(patients)}")
#print(f"Dettaglio pazienti: {dict(patients)}\n")


## PATCHIFICAZIONE MULTITHREADING ##

_thread_state = threading.local()


def init_worker_slide(svs_path, level):
    # One slide handle per worker thread, reused across many patches.
    _thread_state.slide = pyvips.Image.new_from_file(str(svs_path))#, level=level)


def process_single_patch(task):
    """Extract patch at (x, y), filter white background, and save if valid."""
    x, y, output_dir, base_name, patch_size, background_threshold = task
    slide = _thread_state.slide

    patch = slide.crop(x, y, patch_size, patch_size)

    patch_np = np.ndarray(
        buffer=patch.write_to_memory(),
        dtype=np.uint8,
        shape=[patch_size, patch_size, patch.bands]
    )

    white_pixels = np.sum(np.all(patch_np > WHITE_THRESHOLD, axis=2))
    white_ratio = white_pixels / (patch_size * patch_size)

    if white_ratio >= background_threshold:
        return 0

    patch_filename = output_dir / f"{base_name}_{x}_{y}.png"

    if patch.bands == 4:
        patch = patch[:3]

    patch.write_to_file(str(patch_filename) + "[strip]")
    return 1


log_rows = []

max_workers = min(8, os.cpu_count() or 4)

for i, svs_path in tqdm(enumerate(svs_files), total=len(svs_files), desc="Processing SVS files"):

    filename = os.path.basename(svs_path)
    base_name = os.path.splitext(filename)[0]

    print(f"\n=== Processing ({i + 1}/{len(svs_files)}): {base_name} ===")

    bag_name = base_name
    output_dir = PATCHES_ROOT / bag_name
    done_flag = output_dir / "DONE.txt"

    if done_flag.exists():
        print(f"Bag gia elaborata: {bag_name} -> skip")
        continue

    output_dir.mkdir(parents=True, exist_ok=True)

    probe_slide = pyvips.Image.new_from_file(str(svs_path))#, level=LEVEL)
    width = probe_slide.width
    height = probe_slide.height

    x_coords = range(0, width - PATCH_SIZE + 1, PATCH_SIZE)
    y_coords = range(0, height - PATCH_SIZE + 1, PATCH_SIZE)
    tasks = [
        (x, y, output_dir, base_name, PATCH_SIZE, BACKGROUND_THRESHOLD)
        for y in y_coords
        for x in x_coords
    ]

    total_patches = len(tasks)
    total_bag_patches = 0

    with ThreadPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker_slide,
        initargs=(svs_path, None),#, LEVEL),
    ) as executor:
        for saved in tqdm(
            executor.map(process_single_patch, tasks),
            total=total_patches,
            desc=base_name
        ):
            total_bag_patches += saved

    with open(done_flag, "w") as f:
        f.write("COMPLETED")

    print(f"Bag completata: {bag_name} | Patch totali: {total_bag_patches}")

    log_rows.append([
        bag_name,
        total_bag_patches
    ])