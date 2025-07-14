import argparse
import logging
import math
import os 
os.environ['HF_ENDPOINT'] = 'http://hf-mirror.com'
import random
import pandas as pd
import shutil
from pathlib import Path
from wavelet_fix import *
import numpy as np
import PIL
import safetensors
from tqdm.auto import trange, tqdm
from PISA_attn_processor import AttnProcessorDistReciprocal
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from custom_diffusers.pipeline_sdxl import TiledStableDiffusionXLPipeline
from custom_diffusers.attention_processor import AttnProcessor2_0
from dataset import TestDataset
import piq
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
import re
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
# from vqae.autoencoder_kl import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from classical_renderer.scatter import ModuleRenderScatter  # circular aperture
# from classical_renderer.scatter_ex import ModuleRenderScatterEX  # adjustable aperture shape

def swap_words(s: str, x: str, y: str):
    return s.replace(x, chr(0)).replace(y, x).replace(chr(0), y)


logger = get_logger(__name__)


def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
        if hasattr(module, "set_processor") and ('down_blocks.1.attentions.0' in name.lower()):
            if not isinstance(processor, dict):
                module.set_processor(processor)
            else:
                module.set_processor(processor.pop(f"{name}.processor"))

        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

from wavelet_fix import *
import time

@torch.inference_mode()
def log_validation(pipeline,
    seed, train_T,
    accelerator,
    validation_prompt="an excellent photo with a large aperture",
    aif_image=None,disp_coc=None,timeit=False
):
    if timeit:
        t_start = time.time()
    latents = pipeline.vae.encode(aif_image).latent_dist.mode().detach() * pipeline.vae.config.scaling_factor
    generator = None if seed is None else torch.Generator(device=accelerator.device).manual_seed(seed)
    image = pipeline(validation_prompt, height=aif_image.shape[-2], width=aif_image.shape[-1], 
                     generator=generator,
                     guidance_scale=1, return_dict=False,
                     timesteps=[train_T],
                     cross_attention_kwargs={'disp_coc': disp_coc},
                     output_type="pt",
                     latents=latents, 
                     TILE_SIZE=64,  # Default for 24GB VRAM GPU
                     )[0]
    image = Image.fromarray(np.uint8(255*(image[0]).permute(1,2,0).cpu().numpy()))
    if timeit:
        t_end = time.time()
        return (image, t_end - t_start)
    return (image, None)


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencing OSEDiff.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="SG161222/RealVisXL_V5.0",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="small",
        help="Variant of BokehDiff model. Currently we only support model size of `small`.",
        choices=["small"]
    )
    parser.add_argument(
        "--test_data_dir", type=str, default=None, required=True, help="A folder containing the testing data."
    )
    parser.add_argument(
        "--train_T",
        type=int,
        default=499,
        help="Training timestep",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="bokehdiff_outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--K", type=float, default=20, help="Param of the aperture. Larger K -> more blur.")
    parser.add_argument("--upsample", type=float, default=1, help="Perform upsampling on the image before rendering in latent space.")
    parser.add_argument(
        "--data_id",
        type=str,
        default=None,
        help="The folder name of the current inference run",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--organization",
        type=str,
        default="EBB",
        help=("File organization of testing dataset. Should be 'folder', 'pngdepth', or 'EBB'."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.test_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args

def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # Load tokenizer
    tokenizer_1 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_1 = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, use_safetensors=True,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, use_safetensors=True,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision,use_safetensors=True,
    ).eval()
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, use_safetensors=True,
    ).eval()

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    vae.enable_tiling()
    text_encoder_1.eval()
    text_encoder_2.eval()
    logging.disable(logging.CRITICAL) # Filter out warnings

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipeline = TiledStableDiffusionXLPipeline(
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            unet=unet,
            vae=vae,
            scheduler=noise_scheduler,
            add_watermarker=False,
        ).to(weight_dtype)
    pipeline.vae.set_attn_processor(AttnProcessor2_0())
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.unet.set_attn_processor(AttnProcessor2_0())
    val_dataset = TestDataset(args.test_data_dir,
        tokenizer_1=pipeline.tokenizer,
        tokenizer_2=pipeline.tokenizer_2,
        organization = args.organization,
        split="inference",
        upsample=args.upsample,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True
    )

    if args.resume_from_checkpoint:
        if not os.path.isdir(args.resume_from_checkpoint):
            raise NotImplementedError("The specified checkpoint should be a directory!")
        path = args.resume_from_checkpoint
        accelerator.print(f"Resuming from checkpoint {path}")
        pipeline.load_lora_weights(path, 
                    weight_name="pytorch_lora_weights.safetensors")
        pipeline.vae.load_state_dict(
            torch.load(os.path.join(path, "vae.ckpt"), weights_only=True),
            strict=False)
    else:
        print("Using pretrained models from HugggingFace...")
        path = args.resume_from_checkpoint
        accelerator.print(f"Resuming from checkpoint {path}")
        pipeline.load_lora_weights("zcx65535/bokehdiff", variant=args.variant)
        vae_state_dict = torch.hub.load_state_dict_from_url(
            f"http://hf-mirror.com/zcx65535/bokehdiff/resolve/main/{args.variant}/vae.ckpt",
            map_location="cpu", check_hash=True, weights_only=True)
        pipeline.vae.load_state_dict(vae_state_dict, strict=False)

    pipeline.text_encoder, pipeline.text_encoder_2, pipeline.vae, pipeline.unet = accelerator.prepare(pipeline.text_encoder, pipeline.text_encoder_2, pipeline.vae, pipeline.unet)
    pipeline, val_dataloader = accelerator.prepare(pipeline, val_dataloader)

    fn_recursive_attn_processor('unet', pipeline.unet, AttnProcessorDistReciprocal(hard=1e7,supersampling_num=5,segment_num=7,train=False))
    output_dir = os.path.join(args.output_dir, args.data_id)
    os.makedirs(output_dir, exist_ok=True)

    durations = []
    for batch in tqdm(val_dataloader):
        torch.cuda.empty_cache()
        # You may modify this line to create a different focus.
        var = [-0.6*batch['defocus_strength'].min(), 0, -0.6*batch['defocus_strength'].max()]
        for var_cr in var:
            defocus_strength = batch['defocus_strength']+var_cr # shift from the original focal plane
            defocus_st_abs = torch.abs(defocus_strength)
            disparity = batch['disparity']
            defocus_st_abs = defocus_st_abs.to(weight_dtype)
            h, w = defocus_st_abs.shape[-2:]
            amplify = args.K * args.upsample / 10
            image, duration = log_validation(pipeline,
                        args.seed, args.train_T, accelerator,
                        validation_prompt=batch['texts'],
                        aif_image=batch['pixel_values'],
                        disp_coc=torch.cat([disparity, defocus_st_abs*amplify],1).to(weight_dtype),
                        timeit=True)
            durations.append(duration)
            image.resize((int(w/args.upsample), int(h/args.upsample)), resample=Image.Resampling.LANCZOS).save(
                f"{output_dir}/{batch['filename'][0]}_{args.K:.1f}_{var_cr:.2f}.jpg", subsampling=0, quality=100)
    print(f"Average duration: {np.mean(durations):.4f} seconds")

if __name__ == "__main__":
    main()