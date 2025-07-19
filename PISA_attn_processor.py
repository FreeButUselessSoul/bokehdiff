# This module implements the PISA module. @Chengxuan Zhu
import inspect
import math
from importlib import import_module
from typing import Callable, List, Optional, Union
from PIL import Image
from torchvision import transforms as T
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention

logger = logging.get_logger(__name__)

if is_torch_npu_available():
    import torch_npu

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

def customized_scaled_dot_product_attention(query, key, value, weight_matrix=None, disp_coc=None,
                    attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                    hard=1.0, train=True, occ_map=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    
    scale_factor = 1 / math.sqrt(value.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=value.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(value.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    if weight_matrix is None: # Falling to vanilla attention
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias.to(value.device)
        attn_weight_softmax = torch.softmax(attn_weight, dim=-1)
        attn_weight_drop = torch.dropout(attn_weight_softmax, dropout_p, train=train)
        return attn_weight_drop @ value
    else:
        attn_weight_manual = torch.sigmoid(hard*(disp_coc[:,[1],None,:]-weight_matrix))
        attn_weight_qk = query @ key.transpose(-2, -1) * scale_factor 
        attn_weight_qk += attn_bias.to(value.device)
        mask = (occ_map).repeat(1,attn_weight_qk.shape[1],1,1)
        attn_weight_qk_max = torch.max(attn_weight_qk, dim=-2, keepdim=True)[0] # To avoid underflow
        attn_weight_qk = torch.exp(attn_weight_qk-attn_weight_qk_max)
        attn_weight = attn_weight_qk*attn_weight_manual
        attn_weight = attn_weight/((torch.sum(attn_weight, dim=-2, keepdim=True)))
        attn_weight_drop = torch.dropout(attn_weight* (1-mask), dropout_p, train=True)
        return attn_weight_drop.to(value.dtype) @ value
        
class AttnProcessorDistReciprocal:
    r"""
    Processor for implementing scaled dot-product attention, but with respect to circle 
    of confusion and occlusion. 
    Implemented by ZCX at 20240823.
    """
    def __init__(self, hard: float = 3, supersampling_num: int = 4, segment_num: int = 4, train: bool = True):
        '''
        hard: the current "hardness" of the soft step function. Default: 3. should increase as the training goes
        supersampling_num: the number of supersampling per pixel, to get an accurate mask. Default: 4.
        segment_num: the number of sampling along the ray from camera to 3D position. Default: 4.
        '''
        self.supersampling_num = supersampling_num
        self.hard = hard
        self.segment_num = segment_num
        self.train = train
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        shape=None,
        cutoff: torch.Tensor = 51.2,
        disp_coc: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        residual = hidden_states
        enable_dist = (attn.to_k.weight.shape == attn.to_q.weight.shape)
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        len_hidden = int((disp_coc.shape[-1]*disp_coc.shape[-2]/hidden_states.shape[1])**.5)
        shape = [disp_coc.shape[-2]//len_hidden, disp_coc.shape[-1]//len_hidden]
        disp_highres = F.interpolate(disp_coc, size=(shape[0]*3,shape[1]*3), mode='bilinear', align_corners=False)
        disp_coc = F.interpolate(disp_coc, size=shape, mode='bilinear', align_corners=False)
        disp_coc = disp_coc.flatten(2,3)
        index_i, index_j = torch.meshgrid(torch.linspace(-1+1/shape[0],1-1/shape[0],shape[0]), torch.linspace(-1+1/shape[1],1-1/shape[1],shape[1]), indexing="xy")
        index_i = index_i.flatten() # h*w
        index_j = index_j.flatten()
        index_ij = torch.stack([index_i, index_j], dim=-1).to(disp_coc.device) # (h*w, 2)
        if self.segment_num <= 0:
            occ_map = torch.zeros((len(disp_coc),1,len(index_ij),len(index_ij)))
        else:
            occ_map = []
            ps = torch.linspace(.1, .9, self.segment_num).to(disp_coc.device)
            with torch.no_grad():
                for ind,disp in enumerate(disp_highres[:,0]):
                    disp_ravel = disp_coc[ind,[0]][None] # disp_lowres.flatten(-2,-1)[None] # (1, 1, h*w)
                    disp_ps = disp_ravel[0,0][None,:,None]*ps[:,None,None] + \
                            (1-ps[:,None,None]) # sample from the point to the camera (disp=1)
                    P_locs = (((1-disp_ps) * disp_ravel)/(disp_ps*(1-disp_ravel)))[...,None]* \
                            (index_ij[None,None,:,:]-index_ij[None,:,None,:]) + \
                            index_ij[None,:,None,:]
                    if self.supersampling_num > 1:
                        P_locs = torch.cat([
                            P_locs, 
                            (P_locs[None]+(
                                torch.rand(
                                    (self.supersampling_num-1,*P_locs.shape), 
                                    device=P_locs.device
                                )-0.5)*2/shape[0]
                            ).reshape(-1, *P_locs.shape[1:])], 0)
                    actual_disp_ps = F.grid_sample(disp[None,None].repeat(len(P_locs),1,1,1), P_locs, align_corners=False)
                    actual_disp_ps = actual_disp_ps.reshape(self.supersampling_num, -1, 1, *P_locs.shape[1:-1])
                    occ_map.append(torch.mean((((actual_disp_ps - disp_ps[None,:,None,None,:,0] > 0)).sum(axis=1) > 0).float(), 0))
            occ_map = torch.stack(occ_map, dim=0) # (bsz, 1, H, W)
            if self.hard < 1e6:
                occ_map = torch.clamp(occ_map, 0, .99)
        input_ndim = hidden_states.ndim
        if enable_dist:
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                i_ravel, j_ravel = torch.meshgrid(
                    torch.linspace(0,1,height).to(hidden_states.device),
                    torch.linspace(0,1,width).to(hidden_states.device), 
                    indexing="ij")
                
                i_ravel = i_ravel.flatten() # (height*width)
                j_ravel = j_ravel.flatten()
                reci_dist_matrix = torch.sqrt(
                    torch.pow(i_ravel[:, None] - i_ravel[None, :], 2) + 
                    torch.pow(j_ravel[:, None] - j_ravel[None, :], 2)) * cutoff
            else:
                i_ravel, j_ravel = torch.meshgrid(
                    torch.linspace(0,1,shape[0]).to(hidden_states.device), 
                    torch.linspace(0,1,shape[1]).to(hidden_states.device), indexing="ij")
                i_ravel = i_ravel.flatten() # (height*width)
                j_ravel = j_ravel.flatten()
                reci_dist_matrix = torch.sqrt(
                    torch.pow(i_ravel[:, None] - i_ravel[None, :], 2) + 
                    torch.pow(j_ravel[:, None] - j_ravel[None, :], 2)) * cutoff
            disp = disp_coc[:,0] # disp[None, :] is K, disp[:, None] is for arbitrary point
            
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        if enable_dist:
            hidden_states = customized_scaled_dot_product_attention(
                query, key, value, reci_dist_matrix.to(query.device,dtype=query.dtype), disp_coc,
                attn_mask=attention_mask, dropout_p=0.1, is_causal=False, hard=math.log(self.hard), 
                occ_map=occ_map, train=self.train,
            )
            if self.hard < 1e6:
                self.hard += 1 # harder
        else: # vanilla attention
            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
