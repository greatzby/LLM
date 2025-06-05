import argparse
import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import GPTConfig, GPT

def plot_and_save_heatmap(matrix, title="Heatmap", save_dir=".", state_tag="state"):
    """
    Plots a heatmap of the given matrix using matplotlib and saves it to a file.
    
    Parameters:
      - matrix: A numpy array containing the matrix to visualize.
      - title: The title of the plot, which will also be part of the saved filename.
      - save_dir: The directory where the image file will be saved.
      - state_tag: A label representing the current state (e.g., checkpoint name or iteration number) used in the filename.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    
    # Construct the filename (e.g. "10000_ckpt_GPT_First_Block_MLP_c_fc.weight.png")
    filename = f"{state_tag}_{title.replace(' ', '_')}.png"
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path)
    print(f"Saved '{title}' image to: {full_path}")
    plt.close()

def load_model(ckpt_path, device):
    """
    Loads a GPT model from the specified checkpoint path.
    Assumes the checkpoint contains 'model_args' and 'model' entries.
    """
    print("Loading checkpoint:", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Retrieve and construct the model configuration from checkpoint
    config_args = checkpoint["model_args"]
    config = GPTConfig(**config_args)
    print("Model configuration:", config)
    
    # Instantiate the model and load the state dict
    model = GPT(config)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully and set to evaluation mode.")
    return model

def extract_weight_matrices(model):
    """
    Extracts key weight matrices from the model:
      1. From the first Transformer Block's MLP part, extracts c_fc.weight (candidate for W_M').
      2. From the first Transformer Block's Attention part, extracts c_attn.weight (candidate for W_V').
    """
    mlp_weight = model.transformer.h[0].mlp.c_fc.weight.detach().cpu().numpy()
    attn_weight = model.transformer.h[0].attn.c_attn.weight.detach().cpu().numpy()
    return mlp_weight, attn_weight

def main():
    parser = argparse.ArgumentParser(description="Check key weight matrices of the GPT model and save heatmaps to files.")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument('--save_dir', type=str, default=".", help="Directory to save the heatmap images")
    args = parser.parse_args()
    
    device = args.device

    # Use the base name of the checkpoint as the state tag (e.g., extract "10000_ckpt" from "10000_ckpt.pt")
    ckpt_base = os.path.basename(args.ckpt)
    state_tag = os.path.splitext(ckpt_base)[0]

    # Create the save directory if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load the model from the checkpoint
    model = load_model(args.ckpt, device)
    
    # Extract the key weight matrices
    mlp_weight, attn_weight = extract_weight_matrices(model)
    print("Shape of MLP (c_fc.weight) matrix:", mlp_weight.shape)
    print("Shape of Attention (c_attn.weight) matrix:", attn_weight.shape)
    
    # Plot and save the heatmaps with filenames that include the state tag
    plot_and_save_heatmap(mlp_weight,
                          title="GPT_First_Block_MLP_c_fc.weight",
                          save_dir=args.save_dir,
                          state_tag=state_tag)
    plot_and_save_heatmap(attn_weight,
                          title="GPT_First_Block_Attention_c_attn.weight",
                          save_dir=args.save_dir,
                          state_tag=state_tag)

if __name__ == '__main__':
    main()