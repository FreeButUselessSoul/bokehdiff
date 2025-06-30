'''
# --------------------------------------------------------------------------------
#   Color fixed script from Li Yi (https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py)
# --------------------------------------------------------------------------------
'''
import torch
from PIL import Image
from torch import Tensor
import numpy as np
from torch.nn import functional as F
import scipy as sp
import scipy.ndimage
from torchvision.transforms import ToTensor, ToPILImage

def adain_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply adaptive instance normalization
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def wavelet_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply wavelet reconstruction
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def calc_mean_std(feat: Tensor, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat:Tensor, style_feat:Tensor):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output

def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq

def wavelet_reconstruction(content_feat:Tensor, style_feat:Tensor, levels: int = 5):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat, levels)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat, levels)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq


def box(img, r):
    """ O(1) box filter
        img - >= 2d image
        r   - radius of box filter
    """
    (rows, cols) = img.shape[:2]
    imDst = np.zeros_like(img)


    tile = [1] * img.ndim
    tile[0] = r
    imCum = np.cumsum(img, 0)
    imDst[0:r+1, :, ...] = imCum[r:2*r+1, :, ...]
    imDst[r+1:rows-r, :, ...] = imCum[2*r+1:rows, :, ...] - imCum[0:rows-2*r-1, :, ...]
    imDst[rows-r:rows, :, ...] = np.tile(imCum[rows-1:rows, :, ...], tile) - imCum[rows-2*r-1:rows-r-1, :, ...]

    tile = [1] * img.ndim
    tile[1] = r
    imCum = np.cumsum(imDst, 1)
    imDst[:, 0:r+1, ...] = imCum[:, r:2*r+1, ...]
    imDst[:, r+1:cols-r, ...] = imCum[:, 2*r+1 : cols, ...] - imCum[:, 0 : cols-2*r-1, ...]
    imDst[:, cols-r: cols, ...] = np.tile(imCum[:, cols-1:cols, ...], tile) - imCum[:, cols-2*r-1 : cols-r-1, ...]

    return imDst

def _gf_color(I, p, r, eps, s=None):
    """ Color guided filter
    I - guide image (rgb)
    p - filtering input (single channel)
    r - window radius
    eps - regularization (roughly, variance of non-edge noise)
    s - subsampling factor for fast guided filter
    """
    fullI = I
    fullP = p
    if s is not None:
        I = sp.ndimage.zoom(fullI, [1/s, 1/s, 1], order=1)
        p = sp.ndimage.zoom(fullP, [1/s, 1/s], order=1)
        r = round(r / s)

    h, w = p.shape[:2]
    N = box(np.ones((h, w)), r)

    mI_r = box(I[:,:,0], r) / N
    mI_g = box(I[:,:,1], r) / N
    mI_b = box(I[:,:,2], r) / N

    mP = box(p, r) / N

    # mean of I * p
    mIp_r = box(I[:,:,0]*p, r) / N
    mIp_g = box(I[:,:,1]*p, r) / N
    mIp_b = box(I[:,:,2]*p, r) / N

    # per-patch covariance of (I, p)
    covIp_r = mIp_r - mI_r * mP
    covIp_g = mIp_g - mI_g * mP
    covIp_b = mIp_b - mI_b * mP

    # symmetric covariance matrix of I in each patch:
    #       rr rg rb
    #       rg gg gb
    #       rb gb bb
    var_I_rr = box(I[:,:,0] * I[:,:,0], r) / N - mI_r * mI_r
    var_I_rg = box(I[:,:,0] * I[:,:,1], r) / N - mI_r * mI_g
    var_I_rb = box(I[:,:,0] * I[:,:,2], r) / N - mI_r * mI_b

    var_I_gg = box(I[:,:,1] * I[:,:,1], r) / N - mI_g * mI_g
    var_I_gb = box(I[:,:,1] * I[:,:,2], r) / N - mI_g * mI_b

    var_I_bb = box(I[:,:,2] * I[:,:,2], r) / N - mI_b * mI_b

    a = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            sig = np.array([
                [var_I_rr[i,j], var_I_rg[i,j], var_I_rb[i,j]],
                [var_I_rg[i,j], var_I_gg[i,j], var_I_gb[i,j]],
                [var_I_rb[i,j], var_I_gb[i,j], var_I_bb[i,j]]
            ])
            covIp = np.array([covIp_r[i,j], covIp_g[i,j], covIp_b[i,j]])
            a[i,j,:] = np.linalg.solve(sig + eps * np.eye(3), covIp)

    b = mP - a[:,:,0] * mI_r - a[:,:,1] * mI_g - a[:,:,2] * mI_b

    meanA = box(a, r) / N[...,np.newaxis]
    meanB = box(b, r) / N

    if s is not None:
        meanA = sp.ndimage.zoom(meanA, [s, s, 1], order=1)
        meanB = sp.ndimage.zoom(meanB, [s, s], order=1)

    q = np.sum(meanA * fullI, axis=2) + meanB

    return q


def _gf_gray(I, p, r, eps, s=None):
    """ grayscale (fast) guided filter
        I - guide image (1 channel)
        p - filter input (1 channel)
        r - window raidus
        eps - regularization (roughly, allowable variance of non-edge noise)
        s - subsampling factor for fast guided filter
    """
    if s is not None:
        Isub = sp.ndimage.zoom(I, 1/s, order=1)
        Psub = sp.ndimage.zoom(p, 1/s, order=1)
        r = round(r / s)
    else:
        Isub = I
        Psub = p


    (rows, cols) = Isub.shape

    N = box(np.ones([rows, cols]), r)

    meanI = box(Isub, r) / N
    meanP = box(Psub, r) / N
    corrI = box(Isub * Isub, r) / N
    corrIp = box(Isub * Psub, r) / N
    varI = corrI - meanI * meanI
    covIp = corrIp - meanI * meanP


    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = box(a, r) / N
    meanB = box(b, r) / N

    if s is not None:
        meanA = sp.ndimage.zoom(meanA, s, order=1)
        meanB = sp.ndimage.zoom(meanB, s, order=1)

    q = meanA * I + meanB
    return q


def _gf_colorgray(I, p, r, eps, s=None):
    """ automatically choose color or gray guided filter based on I's shape """
    if I.ndim == 2 or I.shape[2] == 1:
        return _gf_gray(I, p, r, eps, s)
    elif I.ndim == 3 and I.shape[2] == 3:
        return _gf_color(I, p, r, eps, s)
    else:
        print("Invalid guide dimensions:", I.shape)


def guided_filter(I, p, r, eps, s=None):
    """ run a guided filter per-channel on filtering input p
        I - guide image (1 or 3 channel)
        p - filter input (n channel)
        r - window raidus
        eps - regularization (roughly, allowable variance of non-edge noise)
        s - subsampling factor for fast guided filter
    """
    if p.ndim == 2:
        p = p[:,:,np.newaxis]

    out = np.zeros_like(p)
    for ch in range(p.shape[2]):
        out[:,:,ch] = _gf_colorgray(I[:,:,ch], p[:,:,ch], r, eps, s)
    return np.squeeze(out) if p.ndim == 2 else out

def sliding_filter(content, color, blocksize=512, deg=3, smooth=8):
    assert isinstance(content, torch.Tensor)
    
    if content.min()<-0.05:
        content = torch.clamp(content*.5+.5, 0, 1)
    if color.min()<-0.05:
        color = torch.clamp(color*.5+.5, 0, 1)
    if content.shape != color.shape:
        color = F.interpolate(color, content.shape[-2:], mode='bilinear', align_corners=False)
    content = content.cpu().detach().numpy()
    color = color.cpu().detach().numpy()
    # original_x = content.flatten(0,1)
    original_x = content.reshape(content.shape[0]*content.shape[1], content.shape[2],content.shape[3])
    
    list_i = np.linspace(-3, 3, blocksize)
    list_j = np.linspace(-3, 3, blocksize)
    gaussian = np.exp(-(np.square(list_i) + np.square(list_j[:,None])) / (2))
    result_y = np.zeros((content.shape[0]*content.shape[1], content.shape[2],content.shape[3]))
    accu_gaussian = np.zeros((content.shape[0]*content.shape[1], content.shape[2],content.shape[3]))
    l1 = list(range(0, content.shape[-2]-blocksize, int(blocksize*0.5)))
    if content.shape[-2]-blocksize not in l1:
        l1.append(content.shape[-2]-blocksize)
    l2 = list(range(0, content.shape[-1]-blocksize, int(blocksize*0.5)))
    if content.shape[-1]-blocksize not in l2:
        l2.append(content.shape[-1]-blocksize)
    meshs = np.meshgrid(np.array(l1,dtype=np.int32), np.array(l2,dtype=np.int32),indexing="ij")
    meshs[0] = meshs[0].flatten()
    meshs[1] = meshs[1].flatten()
    for ind_i, ind_j in zip(*meshs):
        x_val = content[...,ind_i:ind_i+blocksize, ind_j:ind_j+blocksize].reshape(content.shape[0]*content.shape[1], -1)
        y_val = color[...,ind_i:ind_i+blocksize, ind_j:ind_j+blocksize].reshape(content.shape[0]*content.shape[1], -1)
        for i,(x,y) in enumerate(zip(x_val, y_val)):
            ori_x = original_x[i,ind_i:ind_i+blocksize,ind_j:ind_j+blocksize]
            # z1 = np.polyfit(x,y, deg=deg)
            z1 = np.polynomial.polynomial.polyfit(x.flatten(),(y-x).flatten(),deg=1,rcond=1e-10)
            # z1 = T.fit(x, y, deg)
            p1 = np.poly1d(z1)
            result_y[i,ind_i:ind_i+blocksize,ind_j:ind_j+blocksize] += gaussian*(p1(x)+x).reshape(blocksize,blocksize)
            accu_gaussian[i,ind_i:ind_i+blocksize,ind_j:ind_j+blocksize] += gaussian
            import ipdb;ipdb.set_trace()
    result_y /= accu_gaussian
    result_y = np.clip(result_y.reshape(content.shape),0,1)
    return result_y


