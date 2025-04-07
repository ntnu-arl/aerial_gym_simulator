#!/usr/bin/env python3
"""
==============================================================
Side-by-Side Observation Viewer for Original and Decoded Images
==============================================================

This script visualizes side-by-side comparisons of original and
decoded observation images saved during a simulation run.

You can specify either:
- A direct path to a `run_x` folder containing images, OR
- An environment index (env_x) and a mode to choose the first or latest run.

Images will be displayed using OpenCV, with original and decoded images
shown side by side, and each frame shown for a specified duration.

---------------
Usage Examples:
---------------
# View specific run folder
python show_obs.py --folder /tmp/observations/env_0/run_0

# View first run of env_1
python show_obs.py --env 1 --mode first

# View latest run of env_0
python show_obs.py --env 0 --mode latest

# Speed up playback to 50 ms/frame
python show_obs.py --env 0 --mode latest --delay 50

---------------
Arguments:
---------------
--folder     : Full path to the specific run folder (takes priority)
--env        : Index of the environment (e.g., 0, 1, 2...)
--mode       : "first" or "latest" run to visualize for the selected env
--base_path  : Base path where env_x folders are located (default: /tmp)
--delay      : Delay between frames in milliseconds (default: 150)

Press ESC during playback to exit.
"""


import os
import re
import cv2
import argparse
from PIL import Image
import numpy as np

def get_run_folder(base_path: str, env: int, mode: str) -> str:
    """
    Determine the path to the first or latest run for a given environment.
    """
    env_path = os.path.join(base_path, f"env_{env}")
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Environment folder not found: {env_path}")
    
    run_dirs = sorted(
        [d for d in os.listdir(env_path) if d.startswith("run_")],
        key=lambda x: int(x.split("_")[1])
    )
    
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {env_path}")
    
    run_dir = run_dirs[0] if mode == "first" else run_dirs[-1]
    return os.path.join(env_path, run_dir)

def load_frames(folder: str):
    """
    Load original and decoded image paths from the folder.
    """
    pattern_orig = re.compile(r'image(\d+)\.png')
    pattern_dec = re.compile(r'decoded_image(\d+)\.png')
    originals = {}
    decoded = {}

    for file in os.listdir(folder):
        if match := pattern_orig.match(file):
            originals[int(match.group(1))] = os.path.join(folder, file)
        elif match := pattern_dec.match(file):
            decoded[int(match.group(1))] = os.path.join(folder, file)

    common_indices = sorted(set(originals) & set(decoded))
    return [(originals[i], decoded[i]) for i in common_indices]

def play_side_by_side(frames, delay_ms: int = 150):
    """
    Display original and decoded images side-by-side using OpenCV.
    """
    for orig_path, dec_path in frames:
        orig_img = Image.open(orig_path).convert('RGB')
        dec_img = Image.open(dec_path).convert('RGB')

        if orig_img.height != dec_img.height:
            dec_img = dec_img.resize((orig_img.width, orig_img.height))

        combined = Image.new('RGB', (orig_img.width + dec_img.width, orig_img.height))
        combined.paste(orig_img, (0, 0))
        combined.paste(dec_img, (orig_img.width, 0))

        frame = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)
        cv2.imshow('Original | Decoded', frame)

        key = cv2.waitKey(delay_ms)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Visualize side-by-side original and decoded images.")
    parser.add_argument('--folder', type=str, help='Full path to run_x folder containing images.')
    parser.add_argument('--env', type=int, default=0, help='Environment index (e.g., 0, 1, 2...)')
    parser.add_argument('--mode', choices=['first', 'latest'], default="first", help='Whether to load the first or latest run.')
    parser.add_argument('--base_path', type=str, default="/tmp", help='Base folder containing env_x/run_y.')
    parser.add_argument('--delay', type=int, default=150, help='Delay between frames in milliseconds.')

    args = parser.parse_args()

    if args.folder:
        run_folder = args.folder
    elif args.env is not None and args.mode:
        run_folder = get_run_folder(args.base_path, args.env, args.mode)
    else:
        print("❌ You must specify either --folder or both --env and --mode")
        return

    print(f"▶️  Loading from: {run_folder}")
    frames = load_frames(run_folder)

    if not frames:
        print("❌ No valid image pairs found.")
        return

    play_side_by_side(frames, delay_ms=args.delay)

if __name__ == "__main__":
    main()