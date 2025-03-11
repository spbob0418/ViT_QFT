import os
import numpy as np
import torch

def token_select(BS, mode, calibration):
    if mode == "zero": #top 2 frequent pixel 
        prefix_token = torch.zeros(BS, 3, 224, 224)

    elif mode == "one": #top 1 frequent pixel
        prefix_token = torch.ones(BS, 3, 224, 224) 

    elif mode == "random":
        prefix_token = torch.rand(BS, 3, 224, 224)

    elif mode == "background_patch":
        base_dir = "/home/shkim/QT/deit/bg_challenge/prefix_patch"

        if calibration == "raw":
            ###########raw data select ######
            raw_data_select = 0
            ##########raw data select ######
            raw_dir = os.path.join(base_dir, "raw")
            raw_files = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".npy")])

            selected_file = raw_files[raw_data_select]  
            raw_data = np.load(selected_file)
            raw_tensor = torch.tensor(raw_data)  # [3, 224, 224]
            prefix_token = raw_tensor.unsqueeze(0).repeat(BS, 1, 1, 1)
            
        elif calibration == "mean":
            # mean.npy 파일 불러오기
            mean_path = os.path.join(base_dir, "mean", "mean.npy")
            mean_data = np.load(mean_path)
            mean_tensor = torch.tensor(mean_data)  # [3, 224, 224]
            prefix_token = mean_tensor.unsqueeze(0).repeat(BS, 1, 1, 1)

        elif calibration == "median":
            median_path = os.path.join(base_dir, "median", "median.npy")
            median_data = np.load(median_path)
            median_tensor = torch.tensor(median_data)  # [3, 224, 224]

            prefix_token = median_tensor.unsqueeze(0).repeat(BS, 1, 1, 1)

        elif calibration == "gaussian":
            gaussian_path = os.path.join(base_dir, "gaussian", "gaussian.npy")
            gaussian_data = np.load(gaussian_path)
            gaussian_tensor = torch.tensor(gaussian_data)  # [3, 224, 224]

            prefix_token = gaussian_tensor.unsqueeze(0).repeat(BS, 1, 1, 1)

        else:
            raise ValueError(f"Unsupported calibration mode: {calibration}")

    elif mode == "high-frequency":
        # 고주파 패치 생성 (Placeholder)
        raise NotImplementedError("High-Frequency mode is not implemented yet.")

    else:
        raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'zero', 'one', 'background_patch', and 'high-frequency'.")

    return prefix_token
