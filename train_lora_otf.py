# Chengxuan Zhu 2025 All rights reserved.
import argparse
import logging
import math
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import random
import shutil
from pathlib import Path
from functools import partial
from glob import glob

import numpy as np
import re
from safetensors.torch import load_file
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from custom_diffusers.attention_processor import AttnProcessor2_0 # To avoid warnings
from PISA_attn_processor import AttnProcessorDistReciprocal
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from peft import LoraConfig, set_peft_model_state_dict
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,DPMSolverMultistepScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
# from vqae.autoencoder_kl import AutoencoderKL
import cv2
# from diffusers.models.controlnet_xs import ControlNetXSAdapter, UNetControlNetXSModel
# from diffusers.pipelines.controlnet_xs.pipeline_controlnet_xs_sd_xl import StableDiffusionXLControlNetXSPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
import shutil
from diffusers.training_utils import cast_training_params
from dataset import MultiKDataset, OnTheFlyDataset

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------
def swap_words(s: str, x: str, y: str):
    return s.replace(x, chr(0)).replace(y, x).replace(chr(0), y)

def custom_worker_init_fn(worker_id, main_process_accelerator_device):
        """
        Ensures the worker process uses the same GPU as its parent main process.
        And sets seeds if needed.
        """
        if main_process_accelerator_device.type == 'cuda':
            # print(f"Worker {worker_id} (parent rank {os.environ.get('LOCAL_RANK', 'N/A')}): Setting CUDA device to {main_process_accelerator_device.index} (maps to global GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')})")
            torch.cuda.set_device(main_process_accelerator_device.index)

def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
    # if hasattr(module, "set_processor") and ('down_blocks.0' in name.lower()): # ('ctrl' in name.lower() or 'control' in name.lower()) and 
    if hasattr(module, "set_processor") and ('down_blocks.1.attentions.0' in name.lower()): # ('ctrl' in name.lower() or 'control' in name.lower()) and 
    # if hasattr(module, "set_processor") and ('up_blocks.1.attentions.2' in name.lower()): # ('ctrl' in name.lower() or 'control' in name.lower()) and 
        print(">>>>>" , name )
        if not isinstance(processor, dict):
            module.set_processor(processor)
        else:
            module.set_processor(processor.pop(f"{name}.processor"))

    for sub_name, child in module.named_children():
        fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
 
def gamma_correction(image: torch.Tensor, gamma: float = 2.2, eps=1e-7, upper_clip=None):
    if gamma == 1:
        return image
    # sign = torch.sign(image)
    # return sign*(torch.abs(image)+eps)**gamma
    base = image.min().detach()
    span = (image.max()-base).detach()
    image_norm = torch.clamp((image-base)/span,eps,upper_clip).pow(gamma) # normalize to [0,1]
    return image_norm*span + base # denormalize to [base, base+span]
    return 2*(torch.clamp(image*.5+.5,eps,upper_clip).pow(gamma)) - 1

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.30.0.dev0")

logger = get_logger(__name__)

def load_my_TI(pipeline, path:str):
    if os.path.isdir(path):
        files = glob(f'{path}/text_encoder_1-*.safetensors')
        path = sorted(files, key=lambda x: int(x.split(".")[-2].split('-')[-1]))[-1]
    latest_file1 = path
    latest_file2 = path.replace('text_encoder_1', 'text_encoder_2')
    for path, encoder, tokenizer in zip([latest_file1, latest_file2],[pipeline.text_encoder, pipeline.text_encoder_2], [pipeline.tokenizer, pipeline.tokenizer_2]):
        # if path.endswith("safetensors"):
        for keys, weights in load_file(path).items():
            placeholder_tokens = keys.split()
            tokenizer.add_tokens(placeholder_tokens)
            encoder.resize_token_embeddings(len(tokenizer))
            token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
            encoder.get_input_embeddings().weight.data[token_ids] = weights.to(encoder.dtype)
    return
def save_model_card(repo_id: str, images=None, base_model=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# Textual inversion text2image fine-tuning - {repo_id}
These are textual inversion adaption weights for {base_model}. You can find some example images in the following. \n
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
        "textual_inversion",
    ]

    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))



def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="rank of LoRA",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--attn_file", type=str, default="PISA_attn_processor.py", help="Path to the attention processor file.")
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
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--opt_vae", type=int, default=1, help="Whether to optimize the VAE.")
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument('--lr_power', type=float, default=1.0, help='The power to use for the polynomial LR scheduler.')
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=3,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-07, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--ckpt_attn", action="store_true", help="Whether or not to use the attn processor in the checkpoint.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
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
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--lambda_lpips", default=.1, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_gan", default=0.1, type=float)
    parser.add_argument("--gan_step", default=1000, type=int)
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--cv_type", default="clip")
    parser.add_argument("--max_grad_norm",type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction value.")
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=30,
        help=("Max number of checkpoints to store."),
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
    parser.add_argument("--lpips",action="store_true", help="whether to use LPIPS.")
    parser.add_argument("--edge",action="store_true", help="whether to use edge loss")
    parser.add_argument("--scale_coc", action="store_true", help="whether to scale CoC by K strength.")
    parser.add_argument("--ablation1",action="store_true", help="whether to disable MY Attn Processor.")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--set_grads_to_none", action="store_true", help="Whether or not to set gradients to None.")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_aperture_templates_small = [
    "a photo in the aperture of {}",
    "a photo in the aperture of {}",
    "a cropped photo in the aperture of {}",
    "the photo in the aperture of {}",
    "a clean photo in the aperture of {}",
    "a dirty photo in the aperture of {}",
    "a dark photo in the aperture of {}",
    "a photo in the aperture of {}",
    "a cool photo in the aperture of {}",
    "a close-up photo in the aperture of {}",
    "a bright photo in the aperture of {}",
    "a cropped photo in the aperture of {}",
    "a good photo in the aperture of {}",
    "a close-up photo in the aperture of {}",
    "a rendition in the aperture of {}",
    "a nice photo in the aperture of {}",
    "a small photo in the aperture of {}",
    "a weird photo in the aperture of {}",
    "a large photo in the aperture of {}",
] 

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    versions = list(map(lambda x: int(x.split('_')[-1]) if x.startswith('logs') else -1, os.listdir(args.output_dir)))
    if len(versions):
        versions = sorted(versions)[-1]+1
    else:
        versions = 0
    logging_dir = os.path.join(args.output_dir, f'{args.logging_dir}_{versions}')
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    if not (args.resume_from_checkpoint and args.ckpt_attn):
        shutil.copy2('./train_lora_otf.py', args.output_dir)
        # dirname = args.output_dir.strip('/').split('/')[-1]
        # shutil.copy2('./PISA_attn_processor.py', f'ckpt/{dirname}_processor.py')
        
    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(os.path.join(args.output_dir,"img_logs"), exist_ok=True)

        # if args.push_to_hub:
        #     repo_id = create_repo(
        #         repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        #     ).repo_id

    # Load tokenizer
    tokenizer_1 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # noise_scheduler.set_timesteps(1)
    # print(noise_scheduler)
    # noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", use_karras_sigmas=True, local_files_only=True)
    text_encoder_1 = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,subfolder="text_encoder", revision=args.revision,
    ).eval()
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision,
    ).eval()
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, use_safetensors=True # , variant=args.variant
    ).eval()
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, use_safetensors=True, variant=args.variant,
    ).train()

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

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    # unet = UNetControlNetXSModel.from_unet(unet, controlnet)
    weight_dtype = torch.float32
    print(accelerator.mixed_precision)
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    
    pipeline = StableDiffusionXLPipeline(
            # args.pretrained_model_name_or_path,
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            unet=unet,
            vae=vae,
            scheduler=noise_scheduler,
            add_watermarker=False,
        ).to(weight_dtype)
    del text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, unet, vae
    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae.set_attn_processor(AttnProcessor2_0())
    pipeline.unet.set_attn_processor(AttnProcessor2_0())
    if (not args.ablation1) and not (args.ckpt_attn and args.resume_from_checkpoint):
        print(f"Using the attn file from {args.attn_file}!")
        assert os.path.exists(args.attn_file), f"The attn file {args.attn_file} does not exist!"
        from importlib import import_module
        dirname = args.output_dir.strip('/').split('/')[-1]
        shutil.copy2(f'{args.attn_file}', f'ckpt/{dirname}_processor.py')
        model_imported = import_module(f'ckpt.{dirname}_processor')
        fn_recursive_attn_processor('unet', pipeline.unet, getattr(model_imported, 'AttnProcessorDistReciprocal')(supersampling_num=4,segment_num=5))
        print("SET TO My Processor!") 

    # Dataset and DataLoaders creation:
    train_dataset = OnTheFlyDataset(
        data_root=args.train_data_dir,
        size=args.resolution, 
        center_crop=args.center_crop,
        split="train",
        device=accelerator.device,
    )
    if accelerator.num_processes > 1:
        partial_worker_init_fn = partial(custom_worker_init_fn,
                                        main_process_accelerator_device=accelerator.device)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, 
            worker_init_fn=partial_worker_init_fn,
            shuffle=True, num_workers=args.dataloader_num_workers) # , pin_memory=True, persistent_workers=True)
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, 
            shuffle=True, num_workers=args.dataloader_num_workers)#, pin_memory=True, persistent_workers=True)
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "conv1",
            "conv2",
            "conv_shortcut",
            "downsamplers.0.conv",
            "upsamplers.0.conv",
            "time_emb_proj",
        ], # ["to_k", "to_q", "to_v", "to_out.0"]
        # exclude_modules=".*(up_blocks).*",
    )
    for param in pipeline.unet.parameters():
        param.requires_grad_(False)
    pipeline.unet.add_adapter(unet_lora_config)
    lora_layers = list(filter(lambda p: p.requires_grad, pipeline.unet.parameters()))
    print(len(lora_layers))
    for param in pipeline.vae.parameters():
        param.requires_grad_(False)
    if args.opt_vae:
        layers_to_opt = list(pipeline.vae.encoder.conv_in.parameters()) + \
            list(pipeline.vae.encoder.mid_block.parameters())+list(pipeline.vae.encoder.conv_out.parameters())

        for param in layers_to_opt:
            param.requires_grad_(True)

    optimizer = optimizer_class(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters(pipeline.unet))
    print(count_parameters(pipeline.vae))
    if args.opt_vae:
        optimizer_vae = optimizer_class(
            layers_to_opt, #list(pipeline.vae.encoder.parameters()),
            lr=.1*args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        lr_scheduler_vae = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_vae,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # Prepare everything with our `accelerator`.
    scaling_factor = pipeline.vae.config.scaling_factor
    if args.lambda_gan > 0: # add discriminator module
        import vision_aided_loss
        net_disc_TorF = vision_aided_loss.Discriminator(cv_type=args.cv_type,loss_type=args.gan_loss_type, device=accelerator.device)
        net_disc_TorF.requires_grad_(True)
        net_disc_TorF.train()

        optimizer_disc = optimizer_class(
            net_disc_TorF.parameters(),
            lr=args.learning_rate*2,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        lr_scheduler_disc = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer_disc,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=args.max_train_steps * accelerator.num_processes,
                num_cycles=1,
                power=args.lr_power,
            )
        net_disc_TorF, optimizer_disc,lr_scheduler_disc = accelerator.prepare(net_disc_TorF, optimizer_disc, lr_scheduler_disc)
    pipeline.text_encoder, pipeline.text_encoder_2, pipeline.vae, pipeline.unet = accelerator.prepare(
        pipeline.text_encoder, pipeline.text_encoder_2, pipeline.vae, pipeline.unet
    )
    pipeline, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(pipeline, train_dataloader, optimizer, lr_scheduler)
    # pipeline.unet.to(torch.float32)
    pipeline.unet.to(weight_dtype)
    # for param in pipeline.unet.parameters():
    #     if param.requires_grad:
    #         param.data = param.to(torch.float32)
    pipeline.vae.to(torch.float32)
    if args.mixed_precision == "fp16":
        cast_training_params([pipeline.unet], dtype=torch.float32)
    if args.lpips:
        from lpips import LPIPS
        lpips_net = LPIPS(net='vgg').to(accelerator.device)
        for param in lpips_net.parameters():
            param.requires_grad_(False)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("CTRL", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1].split('_')[-1]))
            path = dirs[-1] if len(dirs) > 0 else None
            path = os.path.join(args.output_dir, path)
        else:
            path = args.resume_from_checkpoint

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            with open(os.path.join(path, "pytorch_lora_weights.safetensors"), "rb") as f:
                state_dict = safetensors.torch.load(f.read())
            set_peft_model_state_dict(pipeline.unet, state_dict)
            # pipeline.unet.load_attn_procs(path, weight_name="pytorch_lora_weights.safetensors")
            pipeline.vae.load_state_dict(torch.load(os.path.join(path, "vae.ckpt")))
            global_step = 0 #int(path.split("-")[-1].split('.')[0])
            if args.ckpt_attn:
                print("Using the attn file from ckpt folder!")
                from importlib import import_module
                dirname = path.strip('/').split('/')[-2]
                if not os.path.exists(f'ckpt/{dirname}_processor.py'):
                    shutil.copy2(f'{path}/PISA_attn_processor.py', f'ckpt/{dirname}_processor.py')
                model_imported = import_module(f'ckpt.{dirname}_processor')
                fn_recursive_attn_processor('unet', pipeline.unet, getattr(model_imported, 'AttnProcessorDistReciprocal')(hard=global_step*args.train_batch_size))
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(accelerator.device, dtype=weight_dtype)
    # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
    # and sample from it to get previous sample
    # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
    # snr = (alphas_cumprod / (1 - alphas_cumprod))
    texts = ['an excellent photo with a large aperture']
    encoder_output_2_list = []; encoder_hidden_states_1_list = []; encoder_hidden_states_2_list = []
    for text in texts:
        input_ids_1 = pipeline.tokenizer(text, padding="max_length", truncation=True, max_length=pipeline.tokenizer.model_max_length, return_tensors="pt").input_ids.to(accelerator.device)
        input_ids_2 = pipeline.tokenizer_2(text, padding="max_length", truncation=True, max_length=pipeline.tokenizer_2.model_max_length, return_tensors="pt", ).input_ids.to(accelerator.device)
        encoder_hidden_states_1 = pipeline.text_encoder(input_ids_1, output_hidden_states=True).hidden_states[-2].to(dtype=weight_dtype)
        encoder_output_2 = pipeline.text_encoder_2(
            input_ids_2.reshape(input_ids_1.shape[0], -1), output_hidden_states=True
        )
        encoder_hidden_states_2 = encoder_output_2.hidden_states[-2].to(dtype=weight_dtype)
        encoder_output_2_list.append(encoder_output_2[0])
        encoder_hidden_states_1_list.append(encoder_hidden_states_1)
        encoder_hidden_states_2_list.append(encoder_hidden_states_2)
    encoder_output_2_list = torch.cat(encoder_output_2_list, dim=0)
    encoder_hidden_states_1_list = torch.cat(encoder_hidden_states_1_list, dim=0)
    encoder_hidden_states_2_list = torch.cat(encoder_hidden_states_2_list, dim=0)
    del pipeline.text_encoder, pipeline.text_encoder_2, pipeline.tokenizer, pipeline.tokenizer_2
    init_buffer = {"loss": 0}
    if args.edge:
        init_buffer['lP'] = 0
    if args.lpips:
        init_buffer['lpips'] = 0
    if args.lambda_gan > 0:
        init_buffer['loss_G'] = 0
        init_buffer['loss_D/real'] = 0
        init_buffer['loss_D/fake'] = 0
    buffer = init_buffer
    l_acc = [pipeline.unet,pipeline.vae]
    if args.lambda_gan > 0:
        l_acc += [net_disc_TorF]
    torch.cuda.empty_cache()
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            
            with accelerator.accumulate(*l_acc):
                disparity = batch['disparity']
                defocus_strength = batch['defocus_strength']
                optimizer.zero_grad()
                if accelerator.num_processes > 1:
                    input_latents = pipeline.vae.module.encode(
                        gamma_correction(batch["aif"].to(torch.float32), args.gamma), return_dict=False
                    )[0]
                else:
                    input_latents = pipeline.vae.encode(
                        gamma_correction(batch["aif"].to(torch.float32), args.gamma), return_dict=False
                    )[0]
                # input_latents = input_latents.latent_dist.sample() * scaling_factor
                input_latents = input_latents.mode() * scaling_factor
                with torch.no_grad():
                    bsz = input_latents.shape[0]
                    timesteps = torch.full((bsz,), 499, device=input_latents.device)
                    timesteps = timesteps.long() 
                    # Get the text embedding for conditioning
                    index_for_text = torch.zeros_like(batch['K'],dtype=torch.long)
                    encoder_hidden_states_1 = encoder_hidden_states_1_list[index_for_text]
                    encoder_hidden_states_2 = encoder_hidden_states_2_list[index_for_text]
                    original_size = [
                        (batch["original_size"][0][i].item(), batch["original_size"][1][i].item())
                        for i in range(args.train_batch_size)
                    ]
                    crop_top_left = [
                        (batch["crop_top_left"][0][i].item(), batch["crop_top_left"][1][i].item())
                        for i in range(args.train_batch_size)
                    ]
                    target_size = (args.resolution, args.resolution)
                    add_time_ids = torch.cat([
                            torch.tensor(original_size[i] + crop_top_left[i] + target_size)
                            for i in range(args.train_batch_size)
                        ]).to(accelerator.device, dtype=torch.float32)
                    # added_cond_kwargs = {"text_embeds": encoder_output_2[0], "time_ids": add_time_ids}
                    added_cond_kwargs = {"text_embeds": encoder_output_2_list[index_for_text], "time_ids": add_time_ids}
                    encoder_hidden_states = torch.cat([encoder_hidden_states_1, encoder_hidden_states_2], dim=-1)
                    aif_image, bokeh_image = batch["aif"], batch["pixel_values"]
                    
                disp_coc = torch.cat([disparity,defocus_strength* batch['K'][:,None,None,None].float()],1)
                assert disp_coc.shape[1] == 2
                # energy = torch.clamp(.5*aif_image+.5,0,1)**2.2
                # energy = .2126*energy[:,[0],:,:] + .7152*energy[:,[1],:,:] + .0722*energy[:,[2],:,:]
                model_pred = pipeline.unet(
                    input_latents, timesteps, encoder_hidden_states.to(torch.float32),
                    # controlnet_cond=controlnet_cond,
                    added_cond_kwargs=added_cond_kwargs,
                    cross_attention_kwargs={'disp_coc': disp_coc},
                ).sample
                
                alpha_prod_t = alphas_cumprod[timesteps][:,None,None,None]
                beta_prod_t = 1 - alpha_prod_t
                # # if noise_scheduler.config.prediction_type == "epsilon":
                pred_im = (input_latents - beta_prod_t ** (0.5) * model_pred) / alpha_prod_t ** (0.5)
                if accelerator.num_processes>1:
                    pred_im = pipeline.vae.module.decode(pred_im/scaling_factor, return_dict=False)[0]
                else:
                    pred_im = pipeline.vae.decode(pred_im/scaling_factor, return_dict=False)[0]

                pred_im = gamma_correction(pred_im, 1/args.gamma)

                loss_mse = torch.mean((pred_im-bokeh_image)**2)
                loss = (loss_mse).mean() * args.lambda_l2
                pred_im = torch.clamp(pred_im, -1, 1)
                """
                Multi-scale edge loss: for more details
                """
                if args.edge:
                    nabla_bokeh_1 = bokeh_image[...,:-1] - bokeh_image[...,1:]
                    nabla_bokeh_2 = bokeh_image[...,:-1,:] - bokeh_image[...,1:,:]
                    nabla_aif_1 = aif_image[...,:-1] - aif_image[...,1:]
                    nabla_aif_2 = aif_image[...,:-1,:] - aif_image[...,1:,:]
                    loss_edge = torch.mean(torch.abs(pred_im[...,:-1]-pred_im[...,1:]-nabla_bokeh_1)*((1+torch.maximum(torch.abs(nabla_bokeh_1),torch.abs(nabla_aif_1))))) +\
                        torch.mean(torch.abs(pred_im[...,:-1,:]-pred_im[...,1:,:]-nabla_bokeh_2)*((1+torch.maximum(torch.abs(nabla_bokeh_2),torch.abs(nabla_aif_2)))))
                    for resamp in [2,3]:
                        im_s = F.interpolate(pred_im, scale_factor=1/resamp, mode='area')
                        bokeh_s = F.interpolate(bokeh_image, scale_factor=1/resamp, mode='area')
                        aif_s = F.interpolate(aif_image, scale_factor=1/resamp, mode='area')
                        nabla_bokeh_1 = bokeh_s[...,:-1] - bokeh_s[...,1:]
                        nabla_bokeh_2 = bokeh_s[...,:-1,:] - bokeh_s[...,1:,:]
                        nabla_aif_1 = aif_s[...,:-1] - aif_s[...,1:]
                        nabla_aif_2 = aif_s[...,:-1,:] - aif_s[...,1:,:]
                        # loss_edge += torch.mean(torch.abs(im_s[...,:-1]-im_s[...,1:]-nabla_bokeh_1)) + torch.mean(torch.abs(im_s[...,:-1,:]-im_s[...,1:,:]-nabla_bokeh_2))
                        loss_edge += torch.mean(torch.abs(im_s[...,:-1]-im_s[...,1:]-nabla_bokeh_1)*(1/resamp*(1+torch.maximum(torch.abs(nabla_bokeh_1),torch.abs(nabla_aif_1))))) +\
                            torch.mean(torch.abs(im_s[...,:-1,:]-im_s[...,1:,:]-nabla_bokeh_2)*(1/resamp*(1+torch.maximum(torch.abs(nabla_bokeh_2),torch.abs(nabla_aif_2)))))
                    loss = loss + loss_edge

                """
                LPIPS loss
                """
                if args.lpips:
                    loss_lpips = lpips_net(pred_im, bokeh_image).mean()
                    loss = loss + loss_lpips * args.lambda_lpips
                """
                Generator loss: fool the discriminator
                """
                if args.lambda_gan > 0:
                    if global_step > args.gan_step:
                        net_disc_TorF.eval()
                        lossG = net_disc_TorF(.5+.5*pred_im, for_G=True).mean()
                        loss = loss + lossG * args.lambda_gan
                    else:
                        lossG = net_disc_TorF(.5+.5*pred_im.detach(), for_G=True).mean().detach().item()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.opt_vae:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                    accelerator.clip_grad_norm_(lora_layers, args.max_grad_norm)
                if loss!=loss:
                    print(f"NaN!!! at {step}. Exiting...")
                    exit(0)
                if args.opt_vae:
                    optimizer_vae.step()
                    lr_scheduler_vae.step()
                optimizer.step()
                lr_scheduler.step()
                
                if args.lambda_gan > 0:
                    # if global_step > args.gan_step:
                    net_disc_TorF.train()
                    """
                    Discriminator loss: fake image vs real image
                    """
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    lossD_real = (net_disc_TorF(.5+.5*bokeh_image, for_real=True)).mean()
                    lossD_fake = net_disc_TorF(.5+.5*(pred_im.detach().clamp_(-1,1)), for_real=False).mean()
                    loss_disc = lossD_fake + lossD_real
                    accelerator.backward(loss_disc)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc_TorF.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # images = []
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}", 
                                              "lpips": f"{loss_lpips.item():.3f}" if args.lpips else 0,
                                              "PSNR": f"{-10*torch.log10(loss_mse/4).mean().item():.2f}"})
                    buffer['loss']+=loss_mse.detach().mean().item()
                    if args.edge:
                        buffer['lP']+=loss_edge.detach().item()
                    if args.lpips:
                        buffer['lpips']+=loss_lpips.detach().mean().item()
                    if args.lambda_gan > 0:
                        if global_step > args.gan_step+1:
                            buffer['loss_G'] += lossG.detach().item()
                        else:
                            buffer['loss_G'] += lossG
                        # if global_step > 2001:
                        buffer['loss_D/real'] += lossD_real.detach().item()
                        buffer['loss_D/fake'] += lossD_fake.detach().item()
                        
                    # progress_bar.set_postfix(**buffer)
                    if global_step % 20 == 0:
                        for k,v in buffer.items():
                            buffer[k] = v/20
                        if args.lambda_gan > 0:
                            buffer['loss_D/sum'] = buffer['loss_D/real']+buffer['loss_D/fake']
                        # buffer['loss'] /= 20 ;buffer['lpips'] /=20
                        buffer['lr'] = lr_scheduler.get_last_lr()[0]
                        accelerator.log(buffer, step=global_step)
                        buffer = init_buffer

                    if global_step % args.validation_steps == 1:
                        for bid,(im, larim, aifim) in enumerate(zip(pred_im, bokeh_image, aif_image)):
                            Image.fromarray(np.uint8(255*(torch.cat([im,larim,aifim],-1)*.5+.5).clamp_(0,1).detach().permute(1,2,0).cpu().numpy())).save(f'{args.output_dir}/img_logs/{(global_step//args.validation_steps):04d}_{bid}_PGI.jpg')
                            

                    if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) > args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        # accelerator.save_state(save_path)
                        # Save pipeline.vae
                        if accelerator.num_processes > 1:
                            if args.opt_vae:
                                # torch.save(pipeline.vae.module.state_dict(), f'{save_path}/vae.ckpt')
                                trainable = {}
                                for n, p in pipeline.vae.module.named_parameters():
                                    if p.requires_grad:
                                        trainable[n] = p.detach().cpu()
                                torch.save(trainable, f"{save_path}/vae.ckpt")
                            pipeline.unet.module.save_attn_procs(save_path)
                        else:
                            if args.opt_vae:
                                # torch.save(pipeline.vae.state_dict(), f'{save_path}/vae.ckpt')
                                trainable = {}
                                for n, p in pipeline.vae.named_parameters():
                                    if p.requires_grad:
                                        trainable[n] = p.detach().cpu()
                                torch.save(trainable, f"{save_path}/vae.ckpt")
                            pipeline.unet.save_attn_procs(save_path)
                        logger.info(f"Saved state to {save_path}")

            if global_step >= args.max_train_steps:
                break
        accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()