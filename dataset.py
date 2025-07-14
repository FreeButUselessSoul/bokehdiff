import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import cv2
import argparse
import logging
import math
import random
import shutil
import scipy
from pathlib import Path
import json
from transformers import pipeline
import numpy as np
import re
import copy
import PIL
import safetensors
import torch
import torch.nn.functional as F
from skimage.transform import resize
from itertools import groupby

import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from glob import glob
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from classical_renderer.mpi_multi_reverse import ModuleRenderRT as ModuleRenderReverseRT
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


class OnTheFlyDataset(Dataset):
    def __init__(
        self,
        data_root,
        learnable_property="style",  # [object, style]
        size=512,
        interpolation="bicubic",
        flip_p=0.5,
        split="train",
        center_crop=False,
        dtype=torch.float32,
        device='cuda',
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.split=split
        self.data_root = data_root
        self.learnable_property = learnable_property
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.fg_paths = list(glob(f'{self.data_root}/*fg/*'))
        self.bg_paths = list(glob(f'{self.data_root}/bg/*'))
        self.gamma = 2.2
        self.num_images = len(self.fg_paths)*len(self.bg_paths)
        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation] 
        self.crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        self.gt_renderer_reverse = ModuleRenderReverseRT().to(device)
        self.ratio_min = 0.5
        self.ratio_max = 1.0

    def __len__(self):
        return int(self.num_images)

    def __getitem__(self, i):
        example = {}
        fg_filepath = self.fg_paths[i % len(self.fg_paths)]
        bg_filepath = self.bg_paths[i // len(self.fg_paths)]
        W, H, K = 768, 768, 12
        times = 2**(random.random()*5-2) # 2~64
        fg = np.array(Image.open(fg_filepath).convert("RGBA"))/255.
        bg_image = np.array(Image.open(bg_filepath).convert("RGB").resize((int(W+4*K),int(H+4*K)),Image.Resampling.LANCZOS))/255.
        fg_image = fg[..., :-1]
        fg_alpha = fg[..., -1:]
        index = np.argwhere(fg_alpha > 0)
        y_min, y_max = index[:, 0].min(), index[:, 0].max()  # 把前景物体的bounding box提取出来
        x_min, x_max = index[:, 1].min(), index[:, 1].max()
        fg_image = fg_image[y_min:y_max, x_min:x_max]
        fg_alpha = fg_alpha[y_min:y_max, x_min:x_max]
        h_fg, w_fg, _ = fg_image.shape
        scale_factor = np.random.uniform(self.ratio_min * ((H + 4 * K) * (W + 4 * K) / h_fg / w_fg) ** (1 / 2),
                                         self.ratio_max * ((H + 4 * K) * (W + 4 * K) / h_fg / w_fg) ** (1 / 2))
        h_fg = int(h_fg * scale_factor)
        w_fg = int(w_fg * scale_factor)
        h, w, _ = bg_image.shape

        # set depth of background image
        a_c = np.random.uniform(-1, 1)  # a/c
        b_c = np.random.uniform(-1, 1)  # b/c

        value_max = max((0, a_c * (w - 1), b_c * (h - 1), a_c * (w - 1) + b_c * (h - 1)))
        value_min = min((0, a_c * (w - 1), b_c * (h - 1), a_c * (w - 1) + b_c * (h - 1)))
        value_range = value_max - value_min

        k = np.random.uniform(0, 1 / 4 - 1 / 1000)

        a_c = a_c / value_range * k
        b_c = b_c / value_range * k

        value_max = value_max / value_range * k
        value_min = value_min / value_range * k

        i_c = 1 / 1000 + value_max  # 1/c

        a = a_c / i_c
        b = b_c / i_c
        c = 1 / i_c

        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        bg_depth = c / (1 - a * grid_x - b * grid_y)
        bg_depth = bg_depth[..., None].astype(np.float32)
        bg_disp = 1 / bg_depth

        bg_coff = np.array([a, b, c]).astype(np.float32)
        bg_alpha = np.ones((h, w, 1)).astype(np.float32)
        syn_image = bg_image
        syn_disp = bg_disp
        
        fg_disps = np.sort(np.random.uniform(1 / bg_depth.min(), 1, 1)).astype(np.float32)[0]
        val_min = fg_disps
        val_max = fg_disps
        while 1:
            if h_fg < h and w_fg < w:
                break
            else:
                h_fg = int(0.9 * h_fg)
                w_fg = int(0.9 * w_fg)
        fg_image = cv2.resize(fg_image, (w_fg, h_fg), interpolation=cv2.INTER_LANCZOS4)
        fg_alpha = cv2.resize(fg_alpha, (w_fg, h_fg), interpolation=cv2.INTER_LANCZOS4)[..., None]

        start_y, start_x = int(0.5*H + 2 * K), int(.5*W + 2 * K)
        start_x = start_x - w_fg // 2 + np.random.randint(-W // 4, W // 4 + 1)  # 对初始摆放位置进行随机偏移
        start_y = start_y - h_fg // 2 + np.random.randint(-H // 4, H // 4 + 1)
        start_x = max(0, min(start_x, bg_image.shape[1] - fg_image.shape[1]))
        start_y = max(0, min(start_y, bg_image.shape[0] - fg_image.shape[0]))

        end_x = start_x + fg_image.shape[1]
        end_y = start_y + fg_image.shape[0]

        # Expand the foreground to the size of the background
        fg_image_tmp = np.zeros_like(bg_image)
        fg_image_tmp[start_y:end_y, start_x:end_x] = fg_image
        fg_image = fg_image_tmp
        fg_alpha_tmp = np.zeros_like(bg_depth)
        fg_alpha_tmp[start_y:end_y, start_x:end_x] = fg_alpha
        fg_alpha = fg_alpha_tmp
        syn_image = fg_alpha * fg_image + (1 - fg_alpha) * syn_image  # All-in-focus

        a = np.random.uniform(-1, 1)
        b = np.random.uniform(-1, 1)
        fg_disp = (a * grid_x + b * grid_y).astype(np.float32)[..., None]
        value_max = fg_disp[fg_alpha > 0].max()
        value_min = fg_disp[fg_alpha > 0].min()
        fg_disp = (fg_disp - value_min) / (value_max - value_min) * (val_max - val_min) + val_min
        a = a / (value_max - value_min) * (val_max - val_min)
        b = b / (value_max - value_min) * (val_max - val_min)
        c = -value_min / (value_max - value_min) * (val_max - val_min) + val_min
        a = - a / c
        b = - b / c
        c = 1 / c
        syn_disp = syn_disp * (1 - fg_alpha) + fg_disp * fg_alpha  # Synthetic disparity map
        fg_coff = np.array([[a, b, c]]).astype(np.float32)
        syn_image = syn_image.clip(0, 1)[2*K: -2*K, 2*K: -2*K]
        syn_disp = syn_disp[2*K: -2*K, 2*K: -2*K, 0] 
        images = np.concatenate((fg_image[None], bg_image[None]), axis=0).clip(0, 1)
        alphas = np.concatenate((fg_alpha[None], bg_alpha[None]), axis=0)
        coffs = np.concatenate((fg_coff, bg_coff[None]), axis=0)
        alphas = torch.Tensor(alphas).permute(0, 3, 1, 2)[None].to(self.device) 
        coffs = torch.Tensor(coffs)[None].to(self.device)
        images = torch.Tensor(images).permute(0, 3, 1, 2)[None].to(self.device)
        images_linear = images ** self.gamma
        # You may specify the focal plane here.
        # For BokehDiff trainig, we use foreground mean disparity / background mean disparity
        disp_focus = torch.tensor([random.choice([fg_disp.mean(),bg_disp.mean()])]).to(self.device)[:, None,None,None]
        len_disp_focus = 1 # len(disp_focus)
        depth_focus = 1 / disp_focus
        K_adapt = K / torch.max(disp_focus, 1 - disp_focus) * times
        syn_bokehs_cum, syn_weights_cum = self.gt_renderer_reverse(
            images_linear.repeat(len_disp_focus, 1, 1, 1, 1), 
            alphas.repeat(len_disp_focus, 1, 1, 1, 1) * 1, 
            coffs.repeat(len_disp_focus, 1, 1), K_adapt, depth_focus, 0, int(21*times))
        syn_bokehs_linear = syn_bokehs_cum.clamp_(0,1e10) / syn_weights_cum.clamp(1e-5, 1e10)
        syn_bokehs = syn_bokehs_linear[:, :3] ** (1 / self.gamma)
        syn_bokehs = syn_bokehs.clamp(0, 1)
        syn_bokehs = syn_bokehs[0,...,2*K: -2*K, 2*K: -2*K]
        ratio = random.random()* 1/3 + 2/3 # (2/3 ~ 1)

        K_strength = K * ratio * times
        syn = cv2.resize(syn_image, (syn_bokehs.shape[-1], syn_bokehs.shape[-2]), interpolation=cv2.INTER_LANCZOS4)
        disp = cv2.resize(syn_disp,(syn_bokehs.shape[-1], syn_bokehs.shape[-2]), interpolation=cv2.INTER_LANCZOS4)
        defocus_strength = torch.Tensor((np.abs((disp-disp_focus.item())))[None])
        disp = torch.FloatTensor(disp)[None]
        example["original_size"] = (syn_bokehs.shape[-1], syn_bokehs.shape[-2])
        syn = torch.Tensor(syn).permute(2,0,1)
        if self.center_crop:
            y1 = max(0, int(round((syn_bokehs.shape[-2] - self.size) / 2.0)))
            x1 = max(0, int(round((syn_bokehs.shape[-1] - self.size) / 2.0)))
            img = transforms.functional.crop(syn_bokehs, y1, x1, self.size, self.size)
            syn = transforms.functional.crop(syn, y1, x1, self.size, self.size)
            defocus_strength = defocus_strength[...,y1:(y1 + self.size), x1:(x1 + self.size)]
            disp = disp[...,y1:(y1 + self.size), x1:(x1 + self.size)]
        else:
            y1, x1, h, w = self.crop.get_params(syn_bokehs, (self.size, self.size))
            img = transforms.functional.crop(syn_bokehs, y1, x1, h, w)
            syn = transforms.functional.crop(syn, y1, x1, h, w)
            defocus_strength = defocus_strength[...,y1:(y1 + h), x1:(x1 + w)]
            disp = disp[...,y1:(y1+h),x1:(x1+w)]
        example["crop_top_left"] = (y1, x1)
        if random.random()<0.5:
            img, syn, defocus_strength, disp = img.flip(-1), syn.flip(-1), defocus_strength.flip(-1), disp.flip(-1)
        if random.random()<0.5:
            img, syn, defocus_strength, disp = img.flip(-2), syn.flip(-2), defocus_strength.flip(-2), disp.flip(-2)
        if random.random()<0.75:
            scaling = np.clip(.7+random.random(),0.5,1.1)
            img, syn = torch.clamp(img*(scaling),0,1), torch.clamp((syn)*scaling,0,1)
        assert torch.sum(torch.isnan(img))==0, f"img has nan {torch.sum(torch.isnan(img))}"
        example["pixel_values"] = img*2.0 - 1.0
        example["aif"] = syn * 2.0 - 1.0
        example['disparity'] = disp
        example['defocus_strength'] = defocus_strength
        example['K'] = K_strength/10.
        example['disp_focus'] = disp_focus
        return example

class TestDataset(Dataset):
    def __init__(
        self,
        reg_exp, 
        tokenizer_1,
        tokenizer_2,
        interpolation="lanczos",
        split="train",
        dtype=torch.float32,
        organization='folder', # folder or EBB (original/, depth/, ...)
        upsample=1.0,
        last=0,
        cr_K='bokeh', # current GT folder name (if any)
        ero=0, # Erosion
    ):
        self.dtype = dtype
        self.split = split
        self.ero = ero
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.upsample = upsample
        self.organization = organization
        if organization == 'blb':
            assert os.path.exists(os.path.join(reg_exp, '277'))
            self.blb_K = 4
            self.bok_path = sorted(glob(os.path.join(reg_exp, '*', 'bokeh_0'+str(self.blb_K)+'_*.exr')), key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]) + 1000*int(x.split('/')[-2]))
            self.disp_path = [os.path.join(os.path.dirname(p),'depth.exr') for p in self.bok_path]
            self.image_paths = [os.path.join(os.path.dirname(p),'image.exr') for p in self.bok_path]
            self.json_path = [os.path.join(os.path.dirname(p),'info.json') for p in self.bok_path]
            self.num_images = len(self.image_paths)
            return
        if last<0:
            self.image_paths = sorted(glob(reg_exp), key=lambda x: int(os.path.splitext(os.path.basename(x))[0])) 
            self.image_paths = self.image_paths[last:]
        else:
            self.image_paths = sorted(glob(reg_exp))
        self.bok_path = None
        if organization == 'folder':
            self.disp_path = [(name[:-4]+'.npy').replace('image','depth') for name in self.image_paths]
            self.mask_path = [name.replace('image.jpg','mask_portrait.jpg') for name in self.image_paths]
        elif organization == 'pngdepth':
            self.disp_path = sorted(glob(reg_exp.replace('original','depth') + "depth*.png"))
            assert len(self.disp_path) == len(self.image_paths), "Not match"
            if os.path.exists(os.path.dirname(self.image_paths[0]).replace('original','mask')):
                # self.mask_path = sorted(glob(reg_exp.replace('original/','mask/')))
                self.mask_path = [name.replace('.jpg','.png').replace('original/','mask/') for name in self.image_paths]
        elif organization == 'EBB':
            self.disp_path = [(name[:-4]+'_pred.npy').replace('input/','depth/') for name in self.image_paths]
            #sorted(glob(reg_exp.replace('original/','depth/'))) 
            if os.path.exists(os.path.dirname(self.image_paths[0]).replace('input','mask')):
                # self.mask_path = sorted(glob(reg_exp.replace('original/','mask/')))
                self.mask_path = [name.replace('.jpg','.png').replace('input/','mask/') for name in self.image_paths]
            if os.path.exists(os.path.dirname(self.image_paths[0]).replace('input',cr_K+'/')):
                self.bok_path = [name.replace('input/', cr_K+'/') for name in self.image_paths]
                # assert len(os.listdir(self.bok_path)), f"{self.bok_path} is empty"
            

        # self.aif_paths = sorted(glob(f'{self.data_root}/syn*.png'))
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

    def __len__(self):
        return self.num_images

    def __getitem__(self, i):
        example = dict()
        filepath = self.image_paths[i]

        if self.organization=='blb':
            image = cv2.imread(self.image_paths[i], -1)[..., :3].astype(np.float32) ** (1/2.2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Input RGB and output RGB by default
            example["original_size"] = (image.shape[1], image.shape[0])
            shape_enlarge = (int(image.shape[1]*self.upsample), int(image.shape[0]*self.upsample))
            bokeh_gt = cv2.imread(self.bok_path[i], -1)[..., :3].astype(np.float32) ** (1/2.2)
            bokeh_gt = cv2.cvtColor(bokeh_gt, cv2.COLOR_BGR2RGB)
            # bokeh_gt = cv2.resize(bokeh_gt, shape_enlarge, interpolation=cv2.INTER_LANCZOS4)
            image = cv2.resize(image, shape_enlarge, interpolation=cv2.INTER_LANCZOS4)
            bokeh_gt = torch.from_numpy(bokeh_gt).permute(2, 0, 1).contiguous()
            depth = cv2.imread(self.disp_path[i], -1)[..., 0].astype(np.float32)
            depth = cv2.resize(depth, shape_enlarge, interpolation=cv2.INTER_LINEAR)
            disp = 1/(depth+1e-10)
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous() # (3,H,W)
            disp = torch.from_numpy(disp).unsqueeze(0).contiguous() # (1,H,W)
            df_idx = int(self.bok_path[i].split('/')[-1].split('_')[-1].split('.')[0])
            with open(self.json_path[i], 'r') as file:
                info_data = json.load(file)
                Ks = info_data['blur_parameters']
                focus_distances = info_data['focus_distances']
                disp_focus = 1 / focus_distances[df_idx]
                K = Ks[self.blb_K]
                # defocus_strength = disp_focus * K
            texts = "an excellent photo with a large aperture"
            x1 = y1 = 0
            example["crop_top_left"] = (y1, x1)
            example["input_ids_1"] = self.tokenizer_1(texts, padding="max_length", truncation=True, max_length=self.tokenizer_1.model_max_length, return_tensors="pt").input_ids
            example["input_ids_2"] = self.tokenizer_2(texts, padding="max_length", truncation=True, max_length=self.tokenizer_2.model_max_length, return_tensors="pt", ).input_ids
            example["pixel_values"] = image*2-1
            example['disparity'] = disp
            example['defocus_strength'] = disp - disp_focus
            example['texts'] = texts
            example['filename'] = os.path.dirname(self.bok_path[i]).strip('/').split('/')[-1] + '_' + os.path.basename(self.bok_path[i])[:-4]
            example['disp_focus'] = torch.Tensor([disp_focus])
            example['K'] = torch.Tensor([K])
            example['bokeh_GT'] = bokeh_gt #torch.from_numpy(bokeh_gt).permute(2,0,1).to(self.dtype) # [0,1]

            return example
        image = Image.open(filepath).convert("RGB")
        example['real_orisize'] = [image.width, image.height]
        if os.path.exists(os.path.join(os.path.dirname(filepath),'disp_refine.jpg')):
            disp = np.array(Image.open(filepath.replace('image.jpg','disp_refine.jpg')).resize(image.size, resample=Image.Resampling.LANCZOS).convert("L"))/255.
        else:
            if self.disp_path[i].endswith(".npy"):
                disp = np.load(self.disp_path[i])
                disp = (disp - disp.min()) / (disp.max() - disp.min())
                if 'val300/' not in filepath:
                    de_morphed = cv2.dilate(cv2.erode(disp, np.ones((3,3))), np.ones((3,3)))
                    disp = np.clip((disp - de_morphed.min()) / (de_morphed.max() - de_morphed.min()), 0, 1)
            else:
                disp = Image.open(self.disp_path[i]).convert("L") 
                disp = np.array(disp)/255.
                disp = 1-disp
                disp = (disp - disp.min()) / (disp.max() - disp.min())
        #### calculating focus disparity ####
        if hasattr(self, "mask_path"):
            portrait_mask = np.array(Image.open(self.mask_path[i]).resize((disp.shape[1],disp.shape[0]), resample=Image.Resampling.LANCZOS).convert("L"))/255.
            if portrait_mask.max()==0:
                portrait_mask[int(portrait_mask.shape[0]*0.4):int(portrait_mask.shape[0]*0.6),
                            int(portrait_mask.shape[1]*0.4):int(portrait_mask.shape[1]*0.6),
                            ] = 1
            focused_points = np.median(disp[portrait_mask>0.5])
            img = (np.array(image, dtype=np.float32)/255.)
        elif self.bok_path is not None:
            bokeh_GT = Image.open(self.bok_path[i]) #.resize(image.size, resample=Image.Resampling.LANCZOS)
            bokeh_gt = np.array(bokeh_GT, dtype=np.float32)/255.
            focused_points = cv2.Canny(np.array(image),20,170)/255. * cv2.Canny(np.array(bokeh_GT),20,170)/255.
            if np.sum(focused_points>.5)<2:
                focused_points = cv2.Canny(np.array(image),20,120)/255. * cv2.Canny(np.array(bokeh_GT),20,120)/255.
                if np.sum(focused_points>.5)<1:
                    focused_points = cv2.Canny(np.array(image),20,40)/255. * cv2.Canny(np.array(bokeh_GT),20,40)/255.
            disp = cv2.resize(disp, (bokeh_gt.shape[1], bokeh_gt.shape[0]), cv2.INTER_LANCZOS4)
            if np.sum(focused_points[int(0.4*disp.shape[0]):int(0.6*disp.shape[0]),int(0.4*disp.shape[1]):int(0.6*disp.shape[1])])>1:
                disp_part = disp[int(0.4*disp.shape[0]):int(0.6*disp.shape[0]),int(0.4*disp.shape[1]):int(0.6*disp.shape[1])]
                dil = cv2.dilate(focused_points, np.ones((5,5)))[int(0.4*focused_points.shape[0]):int(0.6*focused_points.shape[0]),int(0.4*focused_points.shape[1]):int(0.6*focused_points.shape[1])]
            elif np.sum(focused_points[int(0.3*disp.shape[0]):int(0.7*disp.shape[0]),int(0.3*disp.shape[1]):int(0.7*disp.shape[1])])>1:
                disp_part = disp[int(0.3*disp.shape[0]):int(0.7*disp.shape[0]),int(0.3*disp.shape[1]):int(0.7*disp.shape[1])]
                dil = cv2.dilate(focused_points, np.ones((5,5)))[int(0.3*focused_points.shape[0]):int(0.7*focused_points.shape[0]),int(0.3*focused_points.shape[1]):int(0.7*focused_points.shape[1])]
            else:
                disp_part = disp
                dil = cv2.dilate(focused_points, np.ones((5,5)))
            if len(disp_part[dil>.5][disp_part[dil>.5]<=np.mean(disp[focused_points>.5])+.05])>5:
                focused_points=np.median(
                    disp_part[dil>.5][disp_part[dil>.5]<=np.mean(disp[focused_points>.5])+.05]
                )
            else:
                focused_points=np.median(
                    disp_part[dil>.5][disp_part[dil>.5]<=np.mean(disp[focused_points>.5])+.1]
                )
            # print(self.disp_path[i], self.image_paths[i], self.bok_path[i], dil.shape, disp_part.shape, disp.shape)
            if focused_points!=focused_points:
                Image.fromarray(np.uint8(255*disp_part)).save('debug.jpg')
            # max_overlap = np.max(focused_points.flatten())
            # focused_points = cv2.resize(focused_points, (disp.shape[1], disp.shape[0]), cv2.INTER_NEAREST)
            # thres = focused_points.max()*0.85
            # # if np.sum(focused_points>thres)<20:
            # #     focused_points = np.median(disp[int(disp.shape[0]*.4):int(disp.shape[0]*.6), int(disp.shape[1]*.4):int(disp.shape[1]*.6)])
            # # else:
            # if np.sum(focused_points>thres)<10:
            #     thres = focused_points.max()*0.7
                
            #     # if np.sum(focused_points>thres)<10:
            #     #     thres = focused_points.max()*0.8
            # focused_points = scipy.stats.mode(disp[focused_points>thres], axis=None)[0]
            # focused_points = np.median(disp[focused_points>thres])
            # if np.sum(focused_points>0.9*max_overlap)<60:
            #     if np.sum(focused_points>max_overlap*0.7)<60:
            #     else:
            #         focused_points = np.median(disp[focused_points>0.5])
            # else:
            #     focused_points = np.median(disp[focused_points>0.7])
            
            # modifier = np.mean(bokeh_gt, (0,1), keepdims=True)/np.clip(np.mean(img, (0,1)), 1e-6, 1)
            # img = np.clip(img * modifier,0,1)
            # modifier = np.mean(img, (0,1), keepdims=True)/np.clip(np.mean(bokeh_gt, (0,1)), 1e-6, 1)
            # # if np.abs(1-np.mean(modifier[0,0,:])) > 0.1:
            # bokeh_gt = np.clip(bokeh_gt * modifier,0,1)
            #### End of calculating focus disparity ####
        else:
            raise NotImplementedError("File structure not supported.")
    
        if "inference"== self.split:
            if self.organization=='EBB':
                image = image.resize(
                        (int(image.width*self.upsample), int(image.height*self.upsample)), resample=Image.Resampling.LANCZOS)
                disp = cv2.resize(disp, (image.width, image.height), cv2.INTER_LANCZOS4)
            else:
                new_height, new_width = image.height, image.width
                if new_height < new_width:
                    new_width = max(1024, new_width)
                    new_height = new_width * image.height/image.width
                else:
                    new_height = max(1024, new_height)
                    new_width = new_height * image.width/image.height
                new_width = ((int(new_width)+7)//8)*8
                new_height = ((int(new_height)+7)//8)*8
                image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        
        if self.ero > 0:
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.ero + 1, 2 * self.ero + 1),
                                            (self.ero, self.ero))
            disp = cv2.erode(disp, element)
        elif self.ero < 0:
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (-2 * self.ero + 1, -2 * self.ero + 1),
                                            (-self.ero, -self.ero))
            disp = cv2.dilate(disp, element)
        
        
        
        # defocus_strength = np.abs(1 - disp / np.mean(disp[focused_points>0.25]))[None]
        defocus_strength = (disp - focused_points)[None] # /(1 - focused_points)
        # defocus_strength = (np.abs(focused_points - disp))[None] # if S1 >> f
        defocus_strength = torch.from_numpy(defocus_strength).to(self.dtype)
        disp = torch.from_numpy(disp)[None]
        # placeholder_string = self.placeholder_token
        texts = "an excellent photo with a large aperture"
        # texts = temp.format("with a focus on the subject and blurred background") #, temp.format("with a sharp focus on eveything")]
        # defocus_strength = F.interpolate(defocus_strength[None], (image.height, image.width), mode='bilinear')[0]
        # disp = F.interpolate(disp[None], (image.height, image.width), mode='bilinear')[0]
        # x1 = y1 = 0
        # if image.height > 1024 and image.width > 1024:
        #     y1 = max(0, int(round((image.height - self.size) / 2.0)))
        #     x1 = max(0, int(round((image.width - self.size) / 2.0)))
        #     image = transforms.functional.crop(image, y1, x1, self.size, self.size)
        #     defocus_strength = defocus_strength[...,y1:(y1 + self.size), x1:(x1 + self.size)]
        #     disp = disp[...,y1:(y1 + self.size), x1:(x1 + self.size)]
        # else:
        x1 = y1 = 0
        
        example["crop_top_left"] = (y1, x1)
        example["input_ids_1"] = self.tokenizer_1(texts, padding="max_length", truncation=True, max_length=self.tokenizer_1.model_max_length, return_tensors="pt").input_ids
        example["input_ids_2"] = self.tokenizer_2(texts, padding="max_length", truncation=True, max_length=self.tokenizer_2.model_max_length, return_tensors="pt", ).input_ids

        # default to score-sde preprocessing
        if image.height % 8 !=0 or image.width % 8 !=0:
            image = image.resize((((int(image.width)+7)//8)*8, ((int(image.height)+7)//8)*8), resample=Image.Resampling.LANCZOS)
            disp = F.interpolate(disp[None], (image.height, image.width), mode='bilinear')[0]
            defocus_strength = F.interpolate(defocus_strength[None], (image.height, image.width), mode='bilinear')[0]

        example["original_size"] = (image.width, image.height)
        img = (np.array(image, dtype=np.float32)/255.) 
        if self.bok_path is not None:
            modifier = np.mean(bokeh_gt, (0,1), keepdims=True)/np.clip(np.mean(img, (0,1)), 1e-6, 1)
            img = np.clip(img * modifier,0,1)

        img_tensor = (img * 2 - 1).astype(np.float32)
        img_tensor = torch.from_numpy(img_tensor).permute(2,0,1).to(self.dtype)
        # syn = (syn / 127.5 - 1.0).astype(np.float32)
        # defocus_strength+=torch.clamp((disp-focused_points)*2,0,1)

        example["pixel_values"] = img_tensor
        example['disparity'] = disp
        example['defocus_strength'] = defocus_strength
        example['texts'] = texts
        example['filename'] = filepath.split('/')[-1][:-4]
        example['disp_focus'] = torch.Tensor([focused_points])
        if self.bok_path is not None:
            # bokeh_gt = cv2.resize(bokeh_gt, img_tensor.shape[-1],img_tensor.shape[-2], cv2.INTER_LANCZOS4)
            example['bokeh_GT'] = torch.from_numpy(bokeh_gt).permute(2,0,1).to(self.dtype) # [0,1]

        return example

if __name__ == '__main__':
    import time
    dataset2 = TestDataset(
        '../mass0723/512/val/')