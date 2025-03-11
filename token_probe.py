import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
from datasets_original import build_transform
from torchvision import datasets
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from PIL import Image
# Probing 함수 정의
def outlier_probing(x, block_num, layer, epoch, iteration):

    x = x.detach().cpu().numpy() if x.is_cuda else x.numpy()

    # 첫 100개의 샘플만 사용
    # x = x[:256, :, :]
    x = np.abs(x)
    

    # Max 및 Median 값 계산 (Row-wise)
    max_values = np.max(x, axis=2)  # Shape: [BS, sequence_length]
    median_values = np.median(x, axis=2)  # Shape: [BS, sequence_length]

    # Max를 Median으로 나눈 몫 계산
    epsilon = 1e-8
    safe_median_values = np.where(median_values == 0, epsilon, median_values)
    ratio_values = max_values / safe_median_values

    # 196개의 Min/Median ratio를 각 sample 데이터별로 내림차순 정렬
    ratio_values = np.abs(ratio_values)
    sorted_ratios_per_sample = np.sort(ratio_values, axis=1)[:, ::-1]  # Shape: [10, sequence_length]

    # CSV 파일 저장
    save_dir = f"./token_probing_results/{layer}"
    os.makedirs(save_dir, exist_ok=True)
    if epoch is not None:
        csv_file_path = os.path.join(save_dir, f"block_{block_num}_layer_{layer}_epoch_{epoch}_iteration_{iteration}_min_median_ratios.csv")
    else: 
        csv_file_path = os.path.join(save_dir, f"block_{block_num}_layer_{layer}_iteration_{iteration}_min_median_ratios.csv")

    # 정렬된 결과를 DataFrame으로 저장
    columns = [f"Col{i+1}" for i in range(197)]
    df = pd.DataFrame(sorted_ratios_per_sample, columns=columns)
    df.to_csv(csv_file_path, index=False)


# def outlier_probing_not_sorted(x, block_num, layer, epoch, iteration):
#     # 확인: Input tensor shape [BS, sequence_length, channel_dim]
#     sequence_len = x.shape[1]

#     x = x.detach().cpu().numpy() if x.is_cuda else x.numpy()

#     x = x[:100, :, :]
#     x = np.abs(x)
    
#     # Max 및 Median 값 계산 (Row-wise)
#     max_values = np.max(x, axis=2)  # Shape: [BS, sequence_length]
#     median_values = np.median(x, axis=2)  # Shape: [BS, sequence_length]

#     # Max를 Median으로 나눈 몫 계산
#     epsilon = 1e-8
#     safe_median_values = np.where(median_values == 0, epsilon, median_values)
#     ratio_values = max_values / safe_median_values


#     ratio_values = np.abs(ratio_values)
#     # ratio_values를 정수형으로 변환
#     ratio_values = ratio_values.astype(int)


#     # CSV 파일 저장
#     save_dir = f"/home/shkim/QT_DeiT_small/reproduce/token_probing_results_not_sorted/{layer}"
#     os.makedirs(save_dir, exist_ok=True)
#     csv_file_path = os.path.join(save_dir, f"block_{block_num}_layer_{layer}_epoch_{epoch}_iteration_{iteration}_min_median_ratios.csv")

#     # 정렬된 결과를 DataFrame으로 저장
#     columns = [f"Col{i+1}" for i in range(sequence_len)]
#     df = pd.DataFrame(ratio_values, columns=columns)
#     df.to_csv(csv_file_path, index=False)


def norm_probing_not_sorted(x, block_num, layer, epoch, iteration):
    sequence_len = x.shape[1]

    # GPU에서 실행되는 경우 numpy로 변환
    x = x.detach().cpu().numpy() if x.is_cuda else x.numpy()
    x = np.abs(x)
    
    # Token Wise 계산
    token_max_values = np.max(x, axis=2)  # Shape: [BS, sequence_length]
    token_median_values = np.median(x, axis=2)  # Shape: [BS, sequence_length]

    # Channel Wise 계산
    channel_max_values = np.max(x, axis=1)  # Shape: [BS, channel_length]
    channel_median_values = np.median(x, axis=1)  # Shape: [BS, channel_length]

    # 저장 디렉토리 생성
    token_save_dir = f"./token_probing_results_not_sorted/{layer}"
    channel_save_dir = f"./channel_probing_results_not_sorted/{layer}"
    os.makedirs(token_save_dir, exist_ok=True)
    os.makedirs(channel_save_dir, exist_ok=True)
    
    # CSV 파일 경로 설정
    def get_csv_path(base_dir, value_type):
        if epoch is not None:
            return os.path.join(base_dir, f"block_{block_num}_layer_{layer}_epoch_{epoch}_iteration_{iteration}_{value_type}.csv")
        else:
            return os.path.join(base_dir, f"block_{block_num}_layer_{layer}_iteration_{iteration}_{value_type}.csv")
    
    # Token-wise 저장
    token_max_csv = get_csv_path(token_save_dir, "token_max")
    token_median_csv = get_csv_path(token_save_dir, "token_median")
    
    pd.DataFrame(token_max_values.astype(int), columns=[f"Col{i+1}" for i in range(sequence_len)]).to_csv(token_max_csv, index=False)
    pd.DataFrame(np.round(token_median_values, 4), columns=[f"Col{i+1}" for i in range(sequence_len)]).to_csv(token_median_csv, index=False)
    
    # Channel-wise 저장
    channel_max_csv = get_csv_path(channel_save_dir, "channel_max")
    channel_median_csv = get_csv_path(channel_save_dir, "channel_median")
    
    pd.DataFrame(channel_max_values.astype(int)).to_csv(channel_max_csv, index=False)
    pd.DataFrame(np.round(channel_median_values, 4)).to_csv(channel_median_csv, index=False)



class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# 사용 예시



hidden_states = {}

def hook_fn(module, input, output):
    hidden_states["11th_block"] = output

def eval_probe(model, iteration, args):
    """
    model: Trained model for evaluation
    hidden_state: Transformer hidden states
    reg_num: Number of register tokens
    iteration: Current iteration number
    """
    reg_num = args.register_num
    accum_steps = args.accum_steps
    sampleData_path = args.sample_data_path
    real_iteration = int(iteration / accum_steps)

    #load 1000 images from sampleData_path
    transform = build_transform(is_train=False, args=args)
    dataset = ImageDataset(sampleData_path, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    
    # 모델을 평가 모드로 설정
    actual_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    actual_model.eval()
    hook = actual_model.blocks.modules_list[11].mlp.fc2.register_forward_hook(hook_fn)
    # Run a single sample through the model
    with torch.no_grad():
        for sample_data in dataloader:
            sample_data = sample_data.to(torch.device("cuda"))
            # print("sample_data",len(sample_data))
            output = actual_model(sample_data)

    # print("Hook Type:", type(hook))

    hook.remove()
    hidden_state = hidden_states["11th_block"]

    # Probing hidden state (11th transformer block output)
    # print(hidden_state.size(0))
    x = hidden_state.detach().cpu().numpy()
    x = np.abs(x)
    
    
    # Token-wise 최대값 계산
    token_wise_max_values = np.max(x, axis=2)  # Shape: [BS, sequence_length]
    
    # Register 및 Patch의 정규화 평균값 계산
    reg_norm_avg = np.mean(token_wise_max_values[:, 1:reg_num+2])  # Register 평균
    patch_norm_avg = np.mean(token_wise_max_values[:, reg_num+2 :])  # Patch 전체 평균
    
    # Patch 정규화 표준편차 및 최대값 계산
    patch_norm_std = np.std(token_wise_max_values[:, reg_num+2 :])  # Patch 표준편차
    patch_norm_max = np.max(token_wise_max_values[:, reg_num+2 :])  # Patch 최대값
    
    ########################################################################################
    # Probing 결과 저장
    save_dir = "./eval_probing_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # CSV 저장 경로
    csv_file_path = os.path.join(save_dir, "eval_probe_results.csv")
    
    # CSV 파일에 데이터 추가 (열: iteration, reg_norm_avg, patch_norm_avg, patch_norm_std, patch_norm_max)
    new_data = pd.DataFrame([[real_iteration, reg_norm_avg, patch_norm_avg, patch_norm_std, patch_norm_max]],
                             columns=["Iteration", "Reg_Norm_Avg", "Patch_Norm_Avg", "Patch_Norm_Std", "Patch_Norm_Max"])
    
    # 기존 데이터가 있으면 추가, 없으면 새 파일 생성
    if os.path.exists(csv_file_path):
        existing_data = pd.read_csv(csv_file_path)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        updated_data = new_data
    
    updated_data.round(2).to_csv(csv_file_path, index=False)









