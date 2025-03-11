    @staticmethod
    def backward(ctx, g_3D):
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
        
        ##기록
        # if ctx.device_id == 0 and ctx.iteration%200 ==0 :
            # a_g_3D_quant = a_g_2D_quant.view(reshape_3D[0],reshape_3D[1],-1)
            # # g_3D_file_name = f'reproduce/probe_report_pertensor_test/gradient_mask_test/g_3D_{ctx.layer_info}.pt'
            # # a_g_2D_quant_file_name = f'reproduce/probe_report_pertensor_test/gradient_mask_test/quant_g_2D_{ctx.layer_info}.pt'
            # # torch.save(g_3D[0], g_3D_file_name)
            # # torch.save(a_g_3D_quant[0], a_g_2D_quant_file_name)
            # g_3D_2D = g_3D[0].cpu().numpy()
            # a_g_3D_quant_2D = a_g_3D_quant[0].cpu().numpy()
            
            # g_3D_csv_file = f'/home/shkim/QT_DeiT_small/reproduce/pertensor_4/g_3D_{ctx.layer_info}_epoch{ctx.epoch}_iter{ctx.iteration}.csv'
            # a_g_2D_quant_csv_file = f'/home/shkim/QT_DeiT_small/reproduce/pertensor_4/quant_g_2D_{ctx.layer_info}_epoch{ctx.epoch}_iter{ctx.iteration}.csv'

            # # 데이터프레임 변환 후 CSV 파일에 저장 (새 파일 생성)
            # pd.DataFrame(g_3D_2D).to_csv(g_3D_csv_file, index=False, header=False)
            # pd.DataFrame(a_g_3D_quant_2D).to_csv(a_g_2D_quant_csv_file, index=False, header=False)

        # weight 기록 
        # if ctx.device_id == 0 and ctx.iteration%200 ==0 and ctx.block_num ==11 :
        #     weight_array = q_w.cpu().numpy()  # Move to CPU and convert to numpy
        #     # Define a filename using epoch and iteration
        #     filename = f"weights_epoch{ctx.epoch}_iter{ctx.iteration}_{ctx.layer_info}.npy"
            
        #     directory = "weights_for_plot"  # Directory to save the gradients
        #     if not os.path.exists(directory):  # Create directory if it doesn't exist
        #         os.makedirs(directory)
            
        #     # Define a filename using epoch and iteration
        #     filename = os.path.join(directory, f"weights_head_epoch{ctx.epoch}_iter{ctx.iteration}_{ctx.layer_info}.npy")
            
        #     # Save the numpy array to a file
        #     np.save(filename, weight_array)

        #     print(f"Saved gradient to {filename}")
        # ################################################

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
        
        if ctx.device_id == 0 and ctx.iteration is not None:
            if ctx.iteration % 10000 == 0 and ctx.layer_info is not None:
                probe(grad_X, block_num=ctx.block_num, layer=ctx.layer_info + 'X_grad_after', epoch=ctx.epoch, iteration=ctx.iteration)
                probe(grad_W, block_num=ctx.block_num, layer=ctx.layer_info + 'W_grad_after', epoch=ctx.epoch, iteration=ctx.iteration)
            
        return None, None, None, None, None, grad_X, grad_W, grad_bias, None, None, None, None