import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import GPTConfig, GPT

def plot_and_save_heatmap(matrix, title="Heatmap", save_dir=".", state_tag="state"):
    """
    Plot a heatmap of the given matrix using matplotlib and save it to a file.
    
    Parameters:
      - matrix: A numpy array that contains the matrix to visualize.
      - title: The title of the plot; it will also be part of the saved filename.
      - save_dir: The directory where the image file will be saved.
      - state_tag: A label representing the current state (e.g., checkpoint name or iteration number).
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    
    filename = f"{state_tag}_{title.replace(' ', '_')}.png"
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path)
    print(f"Saved '{title}' image to: {full_path}")
    plt.close()

def plot_heatmap_with_boxes(matrix, mask, title="Heatmap with Boxes", save_dir=".", state_tag="state"):
    """
    Plot a heatmap and overlay red boxes at positions where mask == 1.
    
    Parameters:
      - matrix: The matrix to be plotted.
      - mask: A binary matrix of the same shape as 'matrix'. A value of 1 indicates an "edge" position.
    """
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    im = ax.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    
    nrows, ncols = matrix.shape
    for i in range(nrows):
        for j in range(ncols):
            if mask[i, j] == 1:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     edgecolor='r', facecolor='none', linewidth=1.5)
                ax.add_patch(rect)
    
    filename = f"{state_tag}_{title.replace(' ', '_')}.png"
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path)
    print(f"Saved '{title}' image with boxes to: {full_path}")
    plt.close()

def load_model(ckpt_path, device):
    """
    Load a GPT model from the specified checkpoint path.
    Assumes the checkpoint contains 'model_args' and 'model'.
    """
    print("Loading checkpoint:", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    config_args = checkpoint["model_args"]
    config = GPTConfig(**config_args)
    print("Model configuration:", config)
    
    model = GPT(config)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully and set to evaluation mode.")
    return model

def extract_weight_matrices(model):
    """
    Extract key weight matrices from the model:
      1. From the first Transformer Block's MLP part, extract c_fc.weight (candidate for W_M).
      2. From the first Transformer Block's Attention part, extract c_attn.weight. This matrix is the 
         concatenated result of Q, K, and V, and we extract the last third (associated with the Value) as the candidate for W_V.
    """
    mlp_weight = model.transformer.h[0].mlp.c_fc.weight.detach().cpu().numpy()
    attn_weight = model.transformer.h[0].attn.c_attn.weight.detach().cpu().numpy()
    return mlp_weight, attn_weight

def process_weights(mlp_weight, attn_weight, n_show=20):
    """
    Process the extracted raw weight matrices to display only the first n_show rows and columns:
      - W_M: Take the first n_show rows and columns of mlp_weight to form W_M'.
      - W_V: For attn_weight (shaped [3*d_model, d_model]), extract the last d_model rows corresponding to the Value,
             then take the first n_show rows and columns to form W_V'.
    
    Returns:
      W_M_prime, W_V_prime
    """
    W_M_prime = mlp_weight[:n_show, :n_show]
    
    # attn_weight usually has shape (3*d_model, d_model). Extract the Value portion (the last d_model rows).
    d_model = attn_weight.shape[1]
    W_V = attn_weight[2 * d_model : 3 * d_model, :]
    W_V_prime = W_V[:n_show, :n_show]
    
    return W_M_prime, W_V_prime

def compute_weight_gap(WM, adj_mask):
    """
    Compute the weight gap, defined as:
      gap = average(weight at edge positions) - average(weight at non-edge positions).
    
    Parameters:
      - WM: Processed weight matrix (e.g., W_M') of the same shape as adj_mask.
      - adj_mask: Binary matrix indicating edge positions (1 for an edge, 0 for no edge).
    """
    if WM.shape != adj_mask.shape:
        raise ValueError("Weight matrix and adjacency mask shapes do not match.")
    
    edge_vals = WM[adj_mask == 1]
    non_edge_vals = WM[adj_mask == 0]
    gap = np.mean(edge_vals) - np.mean(non_edge_vals)
    return gap

def main():
    parser = argparse.ArgumentParser(
        description="Extract and process GPT model weight matrices, display the first 20 rows/columns of W_M and W_V, "
                    "and optionally compute the weight gap based on an adjacency matrix."
    )
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument('--save_dir', type=str, default=".", help="Directory to save the heatmap images")
    parser.add_argument('--adjacency', type=str, default=None,
                        help="Path to an adjacency matrix file (.npy format) for gap computation and overlaying red boxes on W_M'")
    args = parser.parse_args()
    
    device = args.device
    ckpt_base = os.path.basename(args.ckpt)
    state_tag = os.path.splitext(ckpt_base)[0]
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load the model from the checkpoint
    model = load_model(args.ckpt, device)
    
    # Extract raw weight matrices
    mlp_weight, attn_weight = extract_weight_matrices(model)
    print("Shape of MLP (c_fc.weight) matrix:", mlp_weight.shape)
    print("Shape of Attention (c_attn.weight) matrix:", attn_weight.shape)
    
    # Process the weights to obtain W_M' and W_V' (showing only the first 20 rows/columns)
    W_M_prime, W_V_prime = process_weights(mlp_weight, attn_weight, n_show=20)
    print("Shape of W_M' (first 20 rows/columns):", W_M_prime.shape)
    print("Shape of W_V' (first 20 rows/columns):", W_V_prime.shape)
    
    # If an adjacency matrix is provided, load and take the first 20x20 piece to create a mask
    if args.adjacency is not None:
        adj = np.load(args.adjacency)
        adj_mask = adj[:20, :20]
        # Plot heatmap for W_M' with red boxes on positions where adj_mask == 1
        plot_heatmap_with_boxes(W_M_prime, adj_mask,
                                title="W_M_prime_with_adjacency",
                                save_dir=args.save_dir, state_tag=state_tag)
        # Compute and print the weight gap
        gap = compute_weight_gap(W_M_prime, adj_mask)
        print(f"Weight gap for W_M' (edge positions avg - non-edge positions avg): {gap:.4f}")
    else:
        plot_and_save_heatmap(W_M_prime, title="W_M_prime", save_dir=args.save_dir, state_tag=state_tag)
    
    # Plot and save W_V' heatmap (without adjacency overlay)
    plot_and_save_heatmap(W_V_prime, title="W_V_prime", save_dir=args.save_dir, state_tag=state_tag)

if __name__ == '__main__':
    main()