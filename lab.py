import torch
from torch.autograd import gradcheck
import torch.nn as nn
from probe import probe

import torch.nn.functional as F

# class STEFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         return torch.round(input)
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output

# class StraightThroughEstimator(nn.Module):
#     def __init__(self):
#         super(StraightThroughEstimator, self).__init__()
#     def forward(self, x):
#         x = STEFunction.apply(x)
#         return x

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
        # self.round_pass = StraightThroughEstimator()

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
            scale = max_x / self.Qp
            x = x / scale 
            # x = self.round_pass(torch.clamp(x, self.Qn, self.Qp)) 
            x = round_pass(x)
            
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
            x = self.round_pass(x.clamp_(self.Qn, self.Qp))
            
        return x, scale

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
     
        x = x.view(-1, x.size(-1)) #reshape to 2D
        input_quant, s_input_quant = a_qmodule(x)
        weight_quant, s_weight_quant = w_qmodule(w)
        # ctx.save_for_backward = (x, s_input_quant, w, s_weight_quant)
        ctx.input = (x, s_input_quant, w, s_weight_quant)

        # output = torch.matmul(input_quant, weight_quant.t())
        # print(input_quant.size())#10,6
        # print(weight_quant.size())#6,2
        output = torch.matmul(input_quant, weight_quant.t())


        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        s_o = s_weight_quant * s_input_quant 

        return output.view(*ctx.reshape_3D_size[:-1], -1) * s_o


    @staticmethod
    def backward(ctx, g_3D):
        # print(g_3D)
        # if ctx.device_id == 0:
        #     if ctx.iteration % 400 == 0 and ctx.layer_info is not None:
        #         probe(g_3D, block_num=ctx.block_num, layer=ctx.layer_info + 'X_grad_before', epoch=ctx.epoch, iteration=ctx.iteration)
        
        g_2D = g_3D.reshape(-1, g_3D.size(-1)) #reshape to 2D
        grad_X = grad_W = grad_bias = None 
        
        q_x, s_x, q_w, s_w = ctx.input


        #since the mixed precision mode, the gradient flows in fp16
        # q_x = q_x.half() 
        # q_w = q_w.half() 
        
        w_g_qmodule, a_g_qmodule = ctx.g_qmodule
        reshape_3D = ctx.reshape_3D_size


        a_g_2D_quant, a_s_g_2D_quant = a_g_qmodule(g_2D)
        grad_X = torch.matmul(a_g_2D_quant, q_w)
        grad_X = grad_X * a_s_g_2D_quant * s_w 
        grad_X = grad_X.view(reshape_3D[0],reshape_3D[1],-1)

        w_g_2D_quant, w_s_g_2D_quant = w_g_qmodule(g_2D)
        grad_W = torch.matmul(w_g_2D_quant.t(), q_x)
        grad_W = grad_W * w_s_g_2D_quant * s_x
        print("grad_W max", grad_W.max())



        if ctx.has_bias:
            grad_bias = g_2D.sum(dim=0)
        else:
            grad_bias = None
        
        # if ctx.device_id == 0:
        #     if ctx.iteration % 400 == 0 and ctx.layer_info is not None:
        #         probe(grad_X, block_num=ctx.block_num, layer=ctx.layer_info + 'X_grad_after', epoch=ctx.epoch, iteration=ctx.iteration)
        #         probe(grad_W, block_num=ctx.block_num, layer=ctx.layer_info + 'W_grad_after', epoch=ctx.epoch, iteration=ctx.iteration)
            
        return None, None, None, None, None, grad_X, grad_W, grad_bias, None, None, None, None


# Define input tensors for testing
input_tensor = 1 + torch.randn((2,5,6), dtype=torch.double, requires_grad=True)
weight_tensor = 1 + torch.randn((2,6), dtype=torch.double, requires_grad=True)
bias_tensor = 1 + torch.randn((2,), dtype=torch.double, requires_grad=True)

# Make sure to use double precision for gradient checking
input_tensor = input_tensor.to(torch.double)
weight_tensor = weight_tensor.to(torch.double)
bias_tensor = bias_tensor.to(torch.double)

# Instantiate your custom quantizer modules
dummy_quantizer = Quantizer(N_bits=None, type="per_tensor", signed=True, symmetric=True)
# Wrap your function and check gradients
grad_check_passed = gradcheck(
    _quantize_global.apply,
    (0, 0, 0, 0, "Layer_Info", input_tensor, weight_tensor, bias_tensor, dummy_quantizer, dummy_quantizer, dummy_quantizer, dummy_quantizer),
    eps=1e-4,
    atol=1e-3
)

print(f"Gradient check passed: {grad_check_passed}")
