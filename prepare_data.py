import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import numpy as np
from transformers import pipeline
from PIL import Image
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="test_data", help="Root directory")
parser.add_argument("--depth_dirname", type=str, default="depth", help="Depth directory name")
parser.add_argument("--image_dirname", type=str, default="image", help="Image directory name")
parser.add_argument("--mask_dirname", type=str, default="mask", help="Mask directory name")
parser.add_argument("--model_size", type=str, default="Base", help="Model size for Depth-Anything-V2", choices=["Small", "Base", "Large"])
args = parser.parse_args()
assert os.path.exists(args.root), f"Root directory {args.root} does not exist."
image_dir = os.path.join(args.root, args.image_dirname)
assert os.path.exists(image_dir), f"Image directory {image_dir} does not exist."
assert len(os.listdir(image_dir)) > 0, f"Image directory {image_dir} is empty."

depth_dir = os.path.join(args.root, args.depth_dirname)
mask_dir = os.path.join(args.root, args.mask_dirname)
os.makedirs(depth_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)


pipe = pipeline(task="depth-estimation", model=f"depth-anything/Depth-Anything-V2-{args.model_size}-hf")
for impath in tqdm(os.listdir(image_dir)):
    if impath.startswith("."):
        continue
    image = Image.open(os.path.join(image_dir, impath)).convert("RGB")
    depth = pipe(image)["predicted_depth"]
    depth = depth.squeeze().cpu().numpy()
    disparity = 1-((depth - depth.min()) / (depth.max() - depth.min()))
    np.save(os.path.join(depth_dir, impath.replace(".jpg", ".npy")), disparity)

# Predict the salient mask with a pretrained model
