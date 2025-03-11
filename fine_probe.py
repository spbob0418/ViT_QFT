import torch
import numpy as np
import csv
import os
import pdb

def probe(x, block_num, layer: str, epoch, iteration):
    # x: Tensor of shape [BS, sequence_length, channel_dim]
    with torch.no_grad():
        if x.dim() == 2: 
            x = x.unsqueeze(0) 
        BS = x.shape[0]
        sequence_length = x.shape[1]
        channel_dim = x.shape[2]
        num_elements = sequence_length * channel_dim

        top1_values = []
        top3_values = []
        top7_values = []  # Top7
        top10_values = []  # Top10
        top15_values = []  # Top15
        top20_values = []  # Top20
        top25_values = []  # Top25
        top3_indices = []
        top1_percent_values = []
        median_values = []
        sample_means = []  # 새로운 리스트: 각 sample의 평균값 저장

        for i in range(BS):
            sample = x[i]  # Shape [sequence_length, channel_dim]
            flattened = sample.view(-1)  # Flatten to 1D tensor of size sequence_length * channel_dim
            sorted_values, sorted_indices = torch.sort(flattened, descending=True)
            top1_value = sorted_values[0].item()
            top1_values.append(top1_value)

            # Top3
            top3_values_i = sorted_values[:3].cpu().numpy()
            top3_indices_flat = sorted_indices[:3].cpu().numpy()
            
            top3_rows = top3_indices_flat // channel_dim
            top3_cols = top3_indices_flat % channel_dim
            top3_indices_i = list(zip(top3_rows, top3_cols))
            top3_values.append(top3_values_i)
            top3_indices.append(top3_indices_i)

            # Top7, Top10, Top15, Top20, Top25
            top7_values.append(sorted_values[:7].cpu().numpy())
            top10_values.append(sorted_values[:10].cpu().numpy())
            top15_values.append(sorted_values[:15].cpu().numpy())
            top20_values.append(sorted_values[:20].cpu().numpy())
            top25_values.append(sorted_values[:25].cpu().numpy())

            # Top1% elements
            top1_percent_count = max(1, int(np.ceil(num_elements * 0.01)))
            top1_percent = sorted_values[:top1_percent_count].cpu().numpy()
            top1_percent_values.append(top1_percent)

            # Median value
            median = torch.median(flattened).item()
            median_values.append(median)

            # Sample Mean
            sample_mean = torch.abs(flattened).mean().item()
            sample_means.append(sample_mean)

        # Compute mean and standard deviation for each quantity
        top1_mean = np.mean(top1_values)
        top1_std = np.std(top1_values)

        top3_means = [np.mean(vals) for vals in top3_values]
        top3_mean = np.mean(top3_means)
        top3_std = np.std(top3_means)

        top7_mean = np.mean([np.mean(vals) for vals in top7_values])
        top7_std = np.std([np.mean(vals) for vals in top7_values])

        top10_mean = np.mean([np.mean(vals) for vals in top10_values])
        top10_std = np.std([np.mean(vals) for vals in top10_values])

        top15_mean = np.mean([np.mean(vals) for vals in top15_values])
        top15_std = np.std([np.mean(vals) for vals in top15_values])

        top20_mean = np.mean([np.mean(vals) for vals in top20_values])
        top20_std = np.std([np.mean(vals) for vals in top20_values])

        top25_mean = np.mean([np.mean(vals) for vals in top25_values])
        top25_std = np.std([np.mean(vals) for vals in top25_values])

        top1_percent_mean = np.mean([np.mean(vals) for vals in top1_percent_values])
        top1_percent_std = np.std([np.mean(vals) for vals in top1_percent_values])

        median_mean = np.mean(median_values)
        median_std = np.std(median_values)

        sample_mean_all = np.mean(sample_means)  # 모든 sample 평균값의 평균
        sample_min = np.min(sample_means)  # 모든 sample 평균값 중 가장 작은 값
        
    os.makedirs("./probe_report", exist_ok=True)

    probe_result_path = f'./probe_report/probe_result_{layer}.csv'
    top3_indices_path = f'./probe_report/top3_indices_{layer}.csv'

    # 통계 데이터를 각 layer에 맞는 probe_result 파일에 기록
    with open(probe_result_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.path.getsize(probe_result_path) == 0:  # 파일이 비어 있으면 헤더 추가
            writer.writerow(['Epoch', 'Iteration', 'Block', 'Top1 Mean', 'Top1 Std', 'Top3 Mean', 'Top3 Std', 
                            'Top7 Mean', 'Top7 Std', 'Top10 Mean', 'Top10 Std', 'Top15 Mean', 'Top15 Std',
                            'Top20 Mean', 'Top20 Std', 'Top25 Mean', 'Top25 Std',
                            'Top1% Mean', 'Top1% Std', 'Median Mean', 'Median Std', 'Sample Mean (All)', 'Sample Min'])
        # 소수 둘째 자리까지만 기록
        writer.writerow([epoch,  epoch*1251 + iteration, block_num,
                        '{:.8f}'.format(top1_mean), '{:.8f}'.format(top1_std),
                        '{:.8f}'.format(top3_mean), '{:.8f}'.format(top3_std),
                        '{:.8f}'.format(top7_mean), '{:.8f}'.format(top7_std),
                        '{:.8f}'.format(top10_mean), '{:.8f}'.format(top10_std),
                        '{:.8f}'.format(top15_mean), '{:.8f}'.format(top15_std),
                        '{:.8f}'.format(top20_mean), '{:.8f}'.format(top20_std),
                        '{:.8f}'.format(top25_mean), '{:.8f}'.format(top25_std),
                        '{:.8f}'.format(top1_percent_mean), '{:.8f}'.format(top1_percent_std),
                        '{:.8f}'.format(median_mean), '{:.8f}'.format(median_std),
                        '{:.8f}'.format(sample_mean_all), '{:.8f}'.format(sample_min)])

    # 각 배치의 Top3 인덱스를 layer별 top3_indices 파일에 기록
    with open(top3_indices_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.path.getsize(top3_indices_path) == 0:  # 파일이 비어 있으면 헤더 추가
            writer.writerow(['Epoch', 'Iteration', 'Block', 'Sample Index', 'Rank', 'Row Index', 'Channel Index'])
        
        for i, indices in enumerate(top3_indices):
            for rank, (row, col) in enumerate(indices, 1):
                writer.writerow([epoch, epoch*1251 + iteration, block_num, i, rank, row, col])
