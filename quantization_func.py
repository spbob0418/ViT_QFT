
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
            elif self.q_type == 'per_token': #토큰 별 가장 큰 값
                max_x = x.abs().amax(dim=-1, keepdim=True)              
            elif self.q_type == 'per_channel': #채널별 가장 큰 값 
                max_x = x.abs().amax(dim=0, keepdim=True)
                
            max_x = max_x.clamp_(self.minimum_range)
            scale = max_x / self.Qp
            x = x / scale 
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
            x = round_pass(x.clamp_(self.Qn, self.Qp))
        
        if self.N_bits == 4: 
            x = x.to(torch.float16)
            scale = scale.to(torch.float16)
            
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
                 in_features, out_features, abits, bias=True):
        super(Quantized_Linear, self).__init__(in_features, out_features, bias=bias)
        self.weight_quantize_module = weight_quantize_module
        self.act_quantize_module = act_quantize_module
        self.weight_grad_quantize_module = weight_grad_quantize_module
        self.act_grad_quantize_module = act_grad_quantize_module
        self.prefix_qmodule = Quantizer(abits, 'per_token')

    def forward(self, input, block_num, epoch, iteration, device_id, prefix_token_num = None, layer_info = None):
        return _quantize_global.apply(block_num, epoch, iteration, device_id, prefix_token_num, layer_info, input, self.weight, self.bias, self.weight_quantize_module,
                                      self.act_quantize_module, self.weight_grad_quantize_module, self.act_grad_quantize_module, self.prefix_qmodule)
    
class _quantize_global(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block_num, epoch, iteration, device_id, prefix_token_num, layer_info, x, w, bias=None, 
                w_qmodule=None, a_qmodule=None, w_g_qmodule=None, a_g_qmodule=None, prefix_qmodule=None):
        #save for backward
        ctx.block_num = block_num
        ctx.iteration = iteration
        ctx.layer_info = layer_info
        ctx.g_qmodule = a_g_qmodule
        ctx.reshape_3D_size = x.size() # x as 3D 
        ctx.has_bias = bias is not None
        ctx.epoch = epoch
        ctx.device_id=device_id
        
        if device_id == 0 and iteration is not None:
            if iteration % 200 == 0 and layer_info is not None:
                probe(w, block_num=block_num, layer=layer_info + 'weight', epoch=epoch, iteration=iteration)
        
        x_2d = x.view(-1, x.size(-1)).to(torch.float16)  # [batch_size * seq_len, feature_dim]
        w = w.to(torch.float16)
        
        ctx.fullprecision_x = x_2d.detach()
        
        if prefix_token_num == None:
            input_quant, s_input_quant = a_qmodule(x_2d)
            weight_quant, s_weight_quant = w_qmodule(w)
            if isinstance(s_weight_quant, int):
                ctx.weight = (weight_quant.detach(), s_weight_quant)
            else:
                ctx.weight = (weight_quant.detach(), s_weight_quant.detach())
            
            output = torch.matmul(input_quant, weight_quant.t())

            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)

            s_o = s_weight_quant * s_input_quant 
            
            return output.view(*ctx.reshape_3D_size[:-1], -1) * s_o
            
            
        prefix_token = x_2d[: (prefix_token_num + 1) * x.size(0)]  # prefix_token_num + 1 개 추출
        pure_x = x_2d[(prefix_token_num + 1) * x.size(0):]  

        #######################
        #quantization 모듈 새로 짜기 
        q_prefix_token, s_prefix_token = prefix_qmodule(prefix_token) #prefix_token quantization : per-token 
        q_pure_x, s_pure_x = a_qmodule(pure_x)# pure_x quantization :기존
        
        input_quant = torch.cat((q_prefix_token, q_pure_x), dim=0)
        
        s_pure_x = s_pure_x.expand(q_pure_x.shape[0])
        s_pure_x = s_pure_x.unsqueeze(-1)
        
        s_input_quant = torch.cat((s_prefix_token, s_pure_x), dim=0)

        weight_quant, s_weight_quant = w_qmodule(w)
        if isinstance(s_weight_quant, int):
            ctx.weight = (weight_quant.detach(), s_weight_quant)
        else:
            ctx.weight = (weight_quant.detach(), s_weight_quant.detach())
        
        output = torch.matmul(input_quant, weight_quant.t())

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        
        s_o = s_weight_quant * s_input_quant
        
        # if not isinstance(s_o, int):
        #     s_o = s_o.to(torch.float32) #--> Cuda out of memory 
        
        with torch.cuda.amp.autocast(enabled=False):
            output = output * s_o
   
        # output = output.to(torch.float32)
        # assert all(tensor.dtype == torch.float16 for tensor in [x_2d,w, input_quant, weight_quant, output, s_weight_quant, s_input_quant, s_o]), "Warning: One or more tensors are not of dtype torch.float16!"
        
        return output.view(*ctx.reshape_3D_size[:-1], -1) 


    @staticmethod
    def backward(ctx, g_3D):
        g_3D = g_3D.to(torch.float16)
        if ctx.device_id == 0 and ctx.iteration is not None:
            if ctx.iteration % 200 == 0 and ctx.layer_info is not None:
                probe(g_3D, block_num=ctx.block_num, layer=ctx.layer_info + 'X_grad_before', epoch=ctx.epoch, iteration=ctx.iteration)

        g_2D = g_3D.reshape(-1, g_3D.size(-1)) #reshape to 2D
        
        
        grad_X = grad_W = grad_bias = None 
        q_w, s_w = ctx.weight
        fullprecision_x = ctx.fullprecision_x

        a_g_qmodule = ctx.g_qmodule
        reshape_3D = ctx.reshape_3D_size
        a_g_2D_quant, a_s_g_2D_quant = a_g_qmodule(g_2D)
        

        
        grad_X = torch.matmul(a_g_2D_quant, q_w)
        grad_X = grad_X * a_s_g_2D_quant * s_w 

        if ctx.layer_info == 'Head':
            grad_W = torch.matmul(g_2D.t(), fullprecision_x)
        else: 
            fullprecision_x = fullprecision_x.to(torch.float16)
            grad_W = torch.matmul(g_2D.t(), fullprecision_x)
            grad_X = grad_X.view(reshape_3D[0],reshape_3D[1],-1)

        if ctx.has_bias:
            grad_bias = g_2D.sum(dim=0)
        else:
            grad_bias = None
        
        if ctx.device_id == 0 and ctx.iteration is not None:
            if ctx.iteration % 200 == 0 and ctx.layer_info is not None:
                probe(grad_X, block_num=ctx.block_num, layer=ctx.layer_info + 'X_grad_after', epoch=ctx.epoch, iteration=ctx.iteration)
                probe(grad_W, block_num=ctx.block_num, layer=ctx.layer_info + 'W_grad_after', epoch=ctx.epoch, iteration=ctx.iteration)
                
        assert all(tensor.dtype == torch.float16 for tensor in [a_g_2D_quant, q_w, g_2D, fullprecision_x,
                                                                grad_X, grad_W, 
                                                                grad_bias if grad_bias is not None else torch.tensor(0.0, dtype=torch.float16)]), "Warning: One or more tensors are not of dtype torch.float16!"


        return None, None, None, None, None, None, grad_X, grad_W, grad_bias, None, None, None, None, None

        