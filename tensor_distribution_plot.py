import os
import torch
import matplotlib.pyplot as plt

def plot_tensor_distribution(x,num_bit, layer_id, quantization_id, iteration, Quant_state, bins=1000):
    if not isinstance(x, torch.Tensor):
        raise ValueError("Input x must be a PyTorch tensor")
    
    # Flatten the tensor to a 1D array
    flattened_x = x.flatten().cpu().numpy()

    # Ensure the directory exists
    save_dir=f"/home/shkim/QT_DeiT_small/reproduce/dist/{num_bit}/{quantization_id}"
    os.makedirs(save_dir, exist_ok=True)

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(flattened_x, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    plt.title(f"Tensor Element Distribution (Layer {layer_id})", fontsize=16)
    plt.xlabel("Element Value", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the plot to the specified directory
    save_path = os.path.join(save_dir, f"iter_{iteration}_layer_{layer_id}_{Quant_state}_distribution.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Plot saved to: {save_path}")
