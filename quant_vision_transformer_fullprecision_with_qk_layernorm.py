import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model

import numpy as np
from probe import probe
from token_probe import token_probing
import pandas as pd
import os
_logger = logging.getLogger(__name__)
__all__ = ['qt_deit_small_patch16_224']

def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

class Quantizer():
    def __init__(self, N_bits: int, type: str = "per_tensor", signed: bool = True, symmetric: bool = True):
        super().__init__()
        self.N_bits = N_bits
        self.signed = signed
        self.symmetric = symmetric
        self.q_type = type
        self.minimum_range = 1e-6
        
        if self.N_bits is None:
            return 

        if self.signed:
            self.Qn = - 2 ** (self.N_bits - 1)
            self.Qp = 2 ** (self.N_bits - 1) - 1
            
        else:
            self.Qn = 0
            self.Qp = 2 ** self.N_bits - 1

    def __call__(self, x):  
        return self.forward(x)

    def forward(self, x): 
        if self.N_bits is None:
            return x, 1
        if self.symmetric:
            if self.q_type == 'per_tensor': 
                max_x = x.abs().max()
            elif self.q_type == 'per_token': 
                max_x = x.abs().amax(dim=-1, keepdim=True)
            elif self.q_type == 'per_channel': 
                max_x = x.abs().amax(dim=0, keepdim=True)
            max_x = max_x.clamp_(self.minimum_range)
            scale = max_x / self.Qp
            x = x / scale 
            x = round_pass(x)
            # x = torch.round(x)
            
        else: #Asymmetric
            if self.q_type == 'per_tensor': 
                min_x = x.min().detach()
                max_x = x.max().detach()
            elif self.q_type == 'per_token': 
                min_x = x.min(dim=-1, keepdim=True).detach()
                max_x = x.max(dim=-1, keepdim=True).detach()
            elif self.q_type == 'per_channel': 
                min_x = x.min(dim=0, keepdim=True).detach()
                max_x = x.max(dim=0, keepdim=True).detach()
            range_x = (max_x - min_x).detach().clamp_(min=self.minimum_range)
            scale = range_x / (self.Qp - self.Qn)
            zero_point = torch.round((min_x / scale) - self.Qn)
            x = (x / scale) + zero_point
            x = round_pass(x.clamp_(self.Qn, self.Qp))
            
        return x, scale


class QuantAct(nn.Module):
    def __init__(self, 
                 N_bits: int, 
                 type: str , 
                 signed: bool = True, 
                 symmetric: bool = True):
        super(QuantAct, self).__init__()
        self.quantizer = Quantizer(N_bits=N_bits, type = type, signed=signed, symmetric=symmetric)

    def forward(self, x):
        q_x, s_qx = self.quantizer(x)
        return q_x, s_qx

class Quantized_Linear(nn.Linear):
    def __init__(self, weight_quantize_module: Quantizer, act_quantize_module: Quantizer, weight_grad_quantize_module: Quantizer, act_grad_quantize_module: Quantizer,
                 in_features, out_features, bias=True):
        super(Quantized_Linear, self).__init__(in_features, out_features, bias=bias)
        self.weight_quantize_module = weight_quantize_module
        self.act_quantize_module = act_quantize_module
        self.weight_grad_quantize_module = weight_grad_quantize_module
        self.act_grad_quantize_module = act_grad_quantize_module

    def forward(self, input, block_num, epoch, iteration, device_id, layer_info):
        return _quantize_global.apply(block_num, epoch, iteration, device_id, layer_info, input, self.weight, self.bias, self.weight_quantize_module,
                                      self.act_quantize_module, self.weight_grad_quantize_module, self.act_grad_quantize_module)
    
class _quantize_global(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block_num, epoch, iteration, device_id, layer_info, x, w, bias=None, w_qmodule=None, a_qmodule=None, w_g_qmodule=None, a_g_qmodule=None):
        #save for backward
        ctx.block_num = block_num
        ctx.iteration = iteration
        ctx.layer_info = layer_info
        ctx.g_qmodule = w_g_qmodule, a_g_qmodule
        ctx.reshape_3D_size = x.size() # x as 3D 
        ctx.has_bias = bias is not None
        ctx.epoch = epoch
        ctx.device_id=device_id
        if device_id == 0 and iteration is not None:
            if iteration % 10000 == 0 and layer_info is not None:
                probe(w, block_num=block_num, layer=layer_info + 'weight', epoch=epoch, iteration=iteration)

        
        x = x.view(-1, x.size(-1)) #reshape to 2D
        input_quant, s_input_quant = a_qmodule(x)
        weight_quant, s_weight_quant = w_qmodule(w)
        ctx.input = (input_quant, s_input_quant, weight_quant, s_weight_quant)
        
        output = torch.matmul(input_quant, weight_quant.t())

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        s_o = s_weight_quant * s_input_quant 

        return output.view(*ctx.reshape_3D_size[:-1], -1) * s_o


    @staticmethod
    def backward(ctx, g_3D):
        #################################
        if ctx.device_id == 0 and ctx.iteration is not None:
            if ctx.iteration % 10000 == 0 and ctx.layer_info is not None:
                probe(g_3D, block_num=ctx.block_num, layer=ctx.layer_info + 'X_grad_before', epoch=ctx.epoch, iteration=ctx.iteration)

        
        g_2D = g_3D.reshape(-1, g_3D.size(-1)) #reshape to 2D
        grad_X = grad_W = grad_bias = None 
        
        q_x, s_x, q_w, s_w = ctx.input

        #since the mixed precision mode, the gradient flows in fp16
        q_x = q_x.half() 
        q_w = q_w.half()


        w_g_qmodule, a_g_qmodule = ctx.g_qmodule
        reshape_3D = ctx.reshape_3D_size
        a_g_2D_quant, a_s_g_2D_quant = a_g_qmodule(g_2D)

        grad_X = torch.matmul(a_g_2D_quant, q_w)
        grad_X = grad_X * a_s_g_2D_quant * s_w 
        
        if ctx.layer_info != 'Head':
            grad_X = grad_X.view(reshape_3D[0],reshape_3D[1],-1)
        
        w_g_2D_quant, w_s_g_2D_quant = w_g_qmodule(g_2D)
        grad_W = torch.matmul(w_g_2D_quant.t(), q_x)
        grad_W = grad_W * w_s_g_2D_quant * s_x

        if ctx.has_bias:
            grad_bias = g_2D.sum(dim=0)
        else:
            grad_bias = None
        
        ############################
        if ctx.device_id == 0 and ctx.iteration is not None:
            if ctx.iteration % 10000 == 0 and ctx.layer_info is not None:
                probe(grad_X, block_num=ctx.block_num, layer=ctx.layer_info + 'X_grad_after', epoch=ctx.epoch, iteration=ctx.iteration)
                probe(grad_W, block_num=ctx.block_num, layer=ctx.layer_info + 'W_grad_after', epoch=ctx.epoch, iteration=ctx.iteration)
    
        return None, None, None, None, None, grad_X, grad_W, grad_bias, None, None, None, None

class Mlp(nn.Module):
    def __init__(
            self,
            block_num,
            abits, 
            wbits,
            w_gbits, 
            a_gbits,
            in_features,
            hidden_features=None,
            act_layer=False):
        super().__init__()
        self.block_num = block_num
        out_features = in_features
        self.fc1 = Quantized_Linear(
                                weight_quantize_module=Quantizer(wbits, 'per_tensor'), 
                                act_quantize_module=Quantizer(abits, 'per_tensor'), 
                                weight_grad_quantize_module=Quantizer(w_gbits, 'per_tensor'),
                                act_grad_quantize_module=Quantizer(a_gbits, 'per_tensor'),
                                in_features=in_features, 
                                out_features=hidden_features, 
                                bias=True
                                )
        self.act = act_layer()
        self.fc2 = Quantized_Linear(
                                weight_quantize_module=Quantizer(wbits, 'per_tensor'), 
                                act_quantize_module=Quantizer(abits, 'per_tensor'), 
                                weight_grad_quantize_module=Quantizer(w_gbits, 'per_tensor'),
                                act_grad_quantize_module=Quantizer(a_gbits, 'per_tensor'),
                                in_features=hidden_features, 
                                out_features=out_features, 
                                bias=True
                                )

    def forward(self, x, epoch, iteration, device_id):
        #################################
        if device_id == 0 and iteration is not None:
            if iteration % 10000 == 0:
                probe(x, block_num=self.block_num, layer='Input_MLP(fc1)', epoch=epoch, iteration=iteration)

        x = self.fc1(x, self.block_num, epoch, iteration, device_id, layer_info = 'During_MLP(fc1)')
        if device_id == 0 and iteration is not None:
            if iteration % 10000 == 0:
                probe(x, block_num=self.block_num, layer='Output_MLP(fc1)', epoch=epoch, iteration=iteration)
            
        x = self.act(x)

        #################################
        if device_id == 0 and iteration is not None:
            if iteration % 10000 == 0:
                probe(x, block_num=self.block_num, layer='Input_MLP(fc2)', epoch=epoch, iteration=iteration)

        x = self.fc2(x, self.block_num, epoch, iteration, device_id, layer_info = 'During_MLP(fc2)')

        if device_id == 0 and iteration is not None:
            if iteration % 10000 == 0:
                probe(x, block_num=self.block_num, layer='Output_MLP(fc2)', epoch=epoch, iteration=iteration)
        return x

class Attention(nn.Module):
    def __init__(
            self,
            block_num,
            abits, 
            wbits, 
            w_gbits,
            a_gbits,
            dim,
            num_heads,
            qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_num = block_num
        
        self.q_norm = nn.LayerNorm(self.head_dim) 
        self.k_norm = nn.LayerNorm(self.head_dim) 

        self.qkv = Quantized_Linear(
                                    weight_quantize_module=Quantizer(wbits, 'per_tensor'), 
                                    act_quantize_module=Quantizer(abits, 'per_tensor'), 
                                    weight_grad_quantize_module=Quantizer(w_gbits, 'per_tensor'),
                                    act_grad_quantize_module=Quantizer(a_gbits, 'per_tensor'),
                                    in_features=dim, 
                                    out_features=dim * 3, 
                                    bias=qkv_bias
                                    )
        # self.qact2 = QuantAct(abits, 'per_tensor')
        self.proj = Quantized_Linear(
                                weight_quantize_module=Quantizer(wbits, 'per_tensor'), 
                                act_quantize_module=Quantizer(abits, 'per_tensor'), 
                                weight_grad_quantize_module=Quantizer(w_gbits, 'per_tensor'),
                                act_grad_quantize_module=Quantizer(a_gbits, 'per_tensor'),
                                in_features=dim, 
                                out_features=dim, 
                                bias=True
        )
        self.qact3 = QuantAct(abits, 'per_tensor')

    def forward(self, x, epoch, iteration, device_id):
        B, N, C = x.shape
        x = self.qkv(x, self.block_num, epoch, iteration, device_id, layer_info = 'qkv') #quantized input, fp output
        qkv = x.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (BN33)
        q, k, v = qkv.unbind(0) 
        q, k = self.q_norm(q), self.k_norm(k)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        ####################
        if device_id == 0 and iteration is not None:
            if iteration % 10000 == 0:
                probe(attn, block_num=self.block_num, layer='QK_Logit', epoch=epoch, iteration=iteration)

        attn = attn.softmax(dim=-1)

        ######################
        if device_id == 0 and iteration is not None:
            if iteration % 10000 == 0:
                probe(attn, block_num=self.block_num, layer='QK_Softmax', epoch=epoch, iteration=iteration)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        if device_id == 0 and iteration is not None:
            if iteration % 10000 == 0:
                probe(x, block_num=self.block_num, layer='Attention_Logit', epoch=epoch, iteration=iteration)

       

        # x, act_scaling_factor = self.qact2(x)
        x = self.proj(x, self.block_num, epoch, iteration, device_id, layer_info='Attention_proj') #quantized input, fp output
        return x

class Q_Block(nn.Module):
    def __init__(self, abits, wbits, w_gbits, a_gbits, block_num, dim, num_heads, mlp_ratio=4., 
                act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.block_num = block_num
        self.qact1 = QuantAct(abits, 'per_tensor')
        self.attn = Attention(
            block_num,
            abits,
            wbits,
            w_gbits,
            a_gbits,
            dim,
            num_heads=num_heads
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.qact2 = QuantAct(abits, 'per_tensor')
        self.mlp = Mlp(
            block_num,
            abits, 
            wbits, 
            w_gbits, 
            a_gbits,
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )
        
    def forward(self, x, epoch, iteration, device_id):
        residual_1 = x
        x = self.norm1(x)
        x = self.attn(x, epoch, iteration, device_id)
        ##########TODO: Probe #############after attention projection
        if device_id == 0 and iteration is not None:
            if iteration % 10000 == 0:
                probe(x, block_num=self.block_num, layer='Attention_proj', epoch=epoch, iteration=iteration)

        x = residual_1 + x
        residual_2 = x 
        x = self.norm2(x)
        x = self.mlp(x, epoch, iteration, device_id) 
        x = residual_2 + x
        ########################################
        if device_id == 0 and iteration is not None:
            if iteration % 10000 == 0:
                probe(x, block_num=self.block_num, layer='Hidden_State', epoch=epoch, iteration=iteration)
                
        if device_id == 0 and iteration is not None and epoch>=9:
            if iteration % 1250 == 0:
                token_probing(x, block_num=self.block_num, layer='Hidden_State', epoch=epoch, iteration=iteration)
        return x

class CustomSequential(nn.Module):
    def __init__(self, *modules):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x, epoch, iteration, device_id):
        for module in self.modules_list:
            x = module(x, epoch, iteration, device_id)
        return x
    

class lowbit_VisionTransformer(VisionTransformer):
    def __init__(self, abits, wbits, w_gbits, a_gbits,
        patch_size, embed_dim, depth, num_heads, mlp_ratio, qkv_bias,
        norm_layer, **kwargs):
        super().__init__(patch_size=patch_size, 
                         embed_dim=embed_dim, 
                         depth=depth, 
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio, 
                         qkv_bias=qkv_bias,
                        norm_layer=norm_layer, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.blocks = CustomSequential(*[
            Q_Block(abits, wbits, w_gbits, a_gbits, block_num=i, dim=embed_dim,
                    num_heads=num_heads, mlp_ratio=mlp_ratio)
            for i in range(depth)])

        self.head = Quantized_Linear(
                        weight_quantize_module=Quantizer(None, 'per_tensor'), 
                        act_quantize_module=Quantizer(None, 'per_tensor'), 
                        weight_grad_quantize_module=Quantizer(None, 'per_tensor'),
                        act_grad_quantize_module=Quantizer(None, 'per_tensor'),
                        in_features=embed_dim, 
                        out_features=1000, 
                        bias=True
                        )
        
    def forward_features(self, x, epoch, iteration, device_id):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x, epoch, iteration, device_id)
        x = self.norm(x)
        
        return x[:, 0]

    def forward(self, x, epoch=None, iteration=None, device_id=None):
        x = self.forward_features(x, epoch, iteration, device_id)
        x = self.head(x, 100, epoch, iteration, device_id, layer_info='Head')
    
        return x

@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = lowbit_VisionTransformer(
        abits = None, wbits = None, w_gbits = None, a_gbits = None,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

    
#################

# @register_model
# def fourbits_deit_small_patch16_224(pretrained=False, **kwargs):
#     model = lowbit_VisionTransformer(
#         abits = 4, wbits = 4, w_gbits = 4, a_gbits = 4,
#         patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         torch.hub.load_state_dict_from_url(
#             url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
#             map_location="cpu", check_hash=True
#         )
#     return model



@register_model
def threebits_deit_small_patch16_224(pretrained=False, **kwargs):
    model = lowbit_VisionTransformer(
        nbits_w = 3, nbits_a = 3,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
            map_location="cpu", check_hash=True
        )
    return model

@register_model
def twobits_deit_small_patch16_224(pretrained=False, **kwargs):
    model = lowbit_VisionTransformer(
        nbits_w = 2, nbits_a = 2,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
            map_location="cpu", check_hash=True
        )
    return model
