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
parser.add_argument("--image_dirname", type=str, default="input", help="Image directory name")
parser.add_argument("--mask_dirname", type=str, default="mask", help="Mask directory name")
parser.add_argument("--model_size", type=str, default="Base", help="Model size for Depth-Anything-V2", choices=["Small", "Base", "Large"])
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert os.path.exists(args.root), f"Root directory {args.root} does not exist."
image_dir = os.path.join(args.root, args.image_dirname)
assert os.path.exists(image_dir), f"Image directory {image_dir} does not exist."
assert len(os.listdir(image_dir)) > 0, f"Image directory {image_dir} is empty."

depth_dir = os.path.join(args.root, args.depth_dirname)
mask_dir = os.path.join(args.root, args.mask_dirname)
os.makedirs(depth_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
with torch.no_grad():
    # Predict disparity with Depth-Anything-V2
    pipe = pipeline(task="depth-estimation", model=f"depth-anything/Depth-Anything-V2-{args.model_size}-hf")
    for impath in tqdm(os.listdir(image_dir)):
        if impath.startswith("."):
            continue
        image = Image.open(os.path.join(image_dir, impath)).convert("RGB")
        depth = pipe(image)["predicted_depth"]
        depth = depth.squeeze().cpu().numpy()
        disparity = 1-((depth - depth.min()) / (depth.max() - depth.min()))
        np.save(os.path.join(depth_dir, impath.replace(".jpg", "_pred.npy")), disparity)
    del pipe # Free memory

    # Predict the salient mask with BiRefNet
    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms
    birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True).eval().to(device)
    transform_image = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    for impath in tqdm(os.listdir(image_dir)):
        image = Image.open(os.path.join(image_dir, impath)).convert("RGB")
        imsize = image.size
        input_images = transform_image(image).unsqueeze(0).to(device)
        mask = np.uint8(255*birefnet(input_images)[-1].sigmoid().cpu()[0].squeeze().numpy())
        Image.fromarray(mask).resize(imsize, resample=Image.Resampling.LANCZOS).save(os.path.join(mask_dir, impath.replace(".jpg", ".png")))
    del birefnet  # Free memory

print(f"Data preparation completed for {args.root} successfully.")