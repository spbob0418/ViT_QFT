import torch
import timm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
from torchvision import transforms
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

def load_vit_model(model_id, device):
    """Load a ViT model from Hugging Face."""
    model = timm.create_model(model_id, pretrained=True).to(device)
    return model

def load_image_dataset(dataset_path, device):
    """Load and preprocess images from the given dataset path."""
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
    ])

    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
    images = []
    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        images.append(img)

    return torch.stack(images).to(device)

def plot_tensor(tensor, layer_idx, layer_type, save_dir):
    """Save a 3D graph of the tensor's absolute values.

    Args:
        tensor (torch.Tensor): Tensor of shape (tokens, channels).
        layer_idx (int): Index of the layer for labeling the plot.
        layer_type (str): Type of the layer (e.g., "Attention", "MLP").
        save_dir (str): Directory where plots are saved.
    """
    tokens, channels = tensor.shape
    x, y = np.meshgrid(range(tokens), range(channels))
    z = np.abs(tensor.cpu().numpy())  # GPU 텐서를 CPU로 복사한 뒤 NumPy 배열로 변환

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z.T, cmap='viridis')

    ax.set_title(f'Layer {layer_idx} ({layer_type}) Output (|Values|)')
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Channel')
    ax.set_zlabel('Value')

    # Save plot to the directory
    plot_path = os.path.join(save_dir, f"layer_{layer_idx}_{layer_type}.png")
    plt.savefig(plot_path)
    plt.close(fig)  # Close the figure to free up memory

def main(model_id, dataset_path):
    """Main function to load the model, dataset, and save layer-wise tensor plots."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_vit_model(model_id, device)
    model.eval()

    # Load dataset
    dataset = load_image_dataset(dataset_path, device)

    # Pass one sample through the model
    sample = dataset[0].unsqueeze(0)  # Use only one sample

    # Hook to extract layer outputs and types
    def hook_fn(module, input, output):
        layer_outputs.append((output, module._get_name()))  # Save tensor and type

    layer_outputs = []
    hooks = []

    # Register hooks on transformer layers
    for name, module in model.named_modules():
        if 'blocks' in name:  # Target only transformer blocks
            hooks.append(module.register_forward_hook(hook_fn))

    # Run inference
    with torch.no_grad():
        _ = model(sample)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Create output directory
    save_dir = "transformer_layer_outputs"

    # Save each layer's output
    for idx, (layer_output, layer_type) in enumerate(layer_outputs):
        # Assuming layer_output is of shape (1, tokens, channels)
        plot_tensor(layer_output.squeeze(0), idx + 1, layer_type, save_dir)

if __name__ == "__main__":
    MODEL_ID = 'timm/vit_small_patch14_dinov2.lvd142m'
    DATASET_PATH = "/home/shkim/QT_DeiT_small/reproduce/sampled_imagenet_val/one_sample_per_class"

    main(MODEL_ID, DATASET_PATH)
