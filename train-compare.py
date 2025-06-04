import os
import time
import math
import re
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from contextlib import nullcontext
import matplotlib.pyplot as plt

from model import GPTConfig, GPT
from logger import get_logger
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='NanoGPT Training with autoregressive evaluation (新版使用与旧版完全一致的训练循环)')
    parser.add_argument('--dataset', type=str, default='simple_graph', help='Name of the dataset to use')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=120, help='Size of the embeddings')
    parser.add_argument('--max_iters', type=int, default=10000, help='Total number of training iterations')
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes')
    parser.add_argument('--num_of_paths', type=int, default=20, help='Number of paths')
    parser.add_argument('--test_interval', type=int, default=100, help='Interval (in iterations) for evaluation')
    parser.add_argument('--ckpt_iter', type=int, default=10000, help='Checkpoint iteration to load')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation')
    return parser.parse_args()

args = parse_args()
dataset      = args.dataset
n_layer      = args.n_layer
n_head       = args.n_head
n_embd       = args.n_embd
max_iters    = args.max_iters
num_nodes    = args.num_nodes
num_of_paths = args.num_of_paths
test_interval= args.test_interval
ckpt_iter   = args.ckpt_iter
device       = args.device
temperature  = args.temperature

data_dir = os.path.join('data', f'{dataset}/{num_nodes}')
meta_path = os.path.join(data_dir, 'meta.pkl')
print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi, itos  = meta['stoi'], meta['itos']
block_size  = meta['block_size']  # 同时作为生成时的最大 token 数
top_k       = len(itos)           # 设为词表大小，可根据需要调整

out_dir = f'out/{dataset}_{n_layer}_{n_head}_{n_embd}_{num_nodes}'
os.makedirs(out_dir, exist_ok=True)

# 初始化 logger
logger = get_logger(os.path.join(out_dir, "train.log"))

# 基本训练参数
gradient_accumulation_steps = 1    # 若要模拟大 batch 可增大该值
train_batch_size = 1024
val_batch_size   = 64
batch_size       = train_batch_size

# 单 GPU 训练设置
master_process = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
# 设置数据类型——这里使用 bfloat16（你也可以改为 float16 或 float32）
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}['bfloat16']
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.manual_seed(1337)
# 利用 memmap 加载数据（train 与 val）
if num_of_paths == 0:
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data   = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
else:
    train_data = np.memmap(os.path.join(data_dir, f'train_{num_of_paths}.bin'), dtype=np.uint16, mode='r')
    val_data   = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    bs = train_batch_size if split == 'train' else val_batch_size
    data_size = block_size + 1
    ix = torch.randint((len(data) - data_size) // data_size, (bs,)) * data_size
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def test_model():
    """
    使用 teacher forcing 模式在验证集上计算 token 级准确率
    """
    X, Y = get_batch('val')
    with ctx:
        logits, _ = model(X, Y)
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == Y).float().sum().item()
    total = Y.numel()
    return correct / total

def encode(s):
    ss = s.split(" ")
    return [stoi[token] for token in ss if token in stoi]

def decode(l):
    return " ".join([itos[i] for i in l])

@torch.no_grad()
def evaluate_autoregressive():
    """
    使用自回归生成方式对测试集计算准确率
    """
    test_file = os.path.join(data_dir, 'test.txt')
    try:
        with open(test_file, encoding='gbk') as f:
            test_lines = [line.strip() for line in f if line.strip() != ""]
    except FileNotFoundError:
        print(f"Test file {test_file} not found. Skipping autoregressive evaluation.")
        return 0.0
    encoded_texts = [encode(line) for line in test_lines]
    if len(encoded_texts) == 0:
        return 0.0
    batch_size_eval = min(1000, len(encoded_texts))
    max_len = max(len(seq) for seq in encoded_texts[:batch_size_eval])
    padded = [seq + [0]*(max_len - len(seq)) for seq in encoded_texts[:batch_size_eval]]
    test_batch = torch.tensor(padded, dtype=torch.long, device=device)
    generated = model.generate(test_batch, max_new_tokens=block_size, temperature=temperature, top_k=top_k)
    total_tokens = 0
    correct_tokens = 0
    for i in range(batch_size_eval):
        orig_len = len(encoded_texts[i])
        gen_tokens = generated[i].tolist()[orig_len:]
        N = min(orig_len, len(gen_tokens))
        total_tokens += N
        for j in range(N):
            if test_batch[i, j].item() == gen_tokens[j]:
                correct_tokens += 1
    ar_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    return ar_accuracy

# ----- 模型初始化：从头训练（init_from='scratch'）或恢复 checkpoint -----
init_from = 'scratch'  # 可选值：'scratch', 'resume', 'gpt2*'
meta_vocab_size = meta.get('vocab_size', None)
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=False, vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
                  dropout=0.0)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    print(f"Resuming training from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=0.0)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)
model.train()

# 初始化 GradScaler 用于 AMP（对 float16 有用，这里基于 ptdtype 判断）
scaler = torch.cuda.amp.GradScaler(enabled=(ptdtype == torch.float16))

# 优化器设置（两边参数完全一致）
weight_decay   = 1e-1
learning_rate  = 5e-4
beta1, beta2   = 0.9, 0.95
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# 学习率调度参数（与旧版一致）
decay_lr = True
warmup_iters = max_iters // 20
lr_decay_iters = max_iters
min_lr = learning_rate / 10

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# 记录训练动态
train_loss_history  = []
train_iter_history  = []
tf_accuracy_history = []  # teacher forcing 模式准确率
ar_accuracy_history = []  # autoregressive 模式准确率
test_iter_history   = []

eval_interval = max_iters // 10
log_interval  = max_iters // 100
eval_iters    = max_iters // 10
eval_only     = False
always_save_checkpoint = True

t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
iter_num = 0  # 确保 iter_num 从 0 开始（如果恢复训练则已设置）

while True:
    # 设置当前迭代的学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 定期评估 loss 并保存 checkpoint
    if iter_num % eval_interval == 0 and master_process:
        losses = {}
        for split in ['train', 'val']:
            loss_vals = []
            for _ in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    _, loss = model(X, Y)
                loss_vals.append(loss.item())
            losses[split] = np.mean(loss_vals)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < globals().get('best_val_loss', 1e9) or always_save_checkpoint:
            best_val_loss = losses['val']
            checkpoint_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
            }
            ckpt_save_path = os.path.join(out_dir, f'{iter_num}_ckpt.pt') if num_of_paths == 0 else os.path.join(out_dir, f'{iter_num}_ckpt_{num_of_paths}.pt')
            torch.save(checkpoint_dict, ckpt_save_path)
            print(f"Checkpoint saved to {ckpt_save_path}")
            logger.info(f"Checkpoint saved to {ckpt_save_path}")

    # 定期评估 teacher forcing 和 autoregressive 模式准确率
    if iter_num % test_interval == 0 and master_process:
        tf_acc = test_model()  # teacher forcing 模式
        ar_acc = evaluate_autoregressive()  # autoregressive 模式
        tf_accuracy_history.append(tf_acc)
        ar_accuracy_history.append(ar_acc)
        test_iter_history.append(iter_num)
        print(f"Iteration {iter_num}: Teacher Forcing Accuracy = {tf_acc:.4f}, Autoregressive Accuracy = {ar_acc:.4f}")
        logger.info(f"Iteration {iter_num}: Teacher Forcing Accuracy = {tf_acc:.4f}, Autoregressive Accuracy = {ar_acc:.4f}")

    if iter_num == 0 and eval_only:
        break

    # 前向-反向更新：使用梯度累积并结合 AMP GradScaler（与旧版完全一致）
    for micro_step in range(gradient_accumulation_steps):
        # 在 AMP 上下文中获取一批数据并计算 loss（记得除以累积步数）
        X, Y = get_batch('train')
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
    # 梯度裁剪
    grad_clip = 1.0
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # 记录时间与 loss
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    loss_item = loss.item() * gradient_accumulation_steps  # 恢复未缩放前的 loss
    train_loss_history.append(loss_item)
    train_iter_history.append(iter_num)
    print(f"iter {iter_num}: loss {loss_item:.4f}, time {dt*1000:.2f}ms")
    logger.info(f"iter {iter_num}: loss {loss_item:.4f}")

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

# 绘制训练曲线和测试准确率曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_iter_history, train_loss_history, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.title('Training Loss Curve')
plt.subplot(1, 2, 2)
plt.plot(test_iter_history, tf_accuracy_history, marker='o', label='Teacher Forcing')
plt.plot(test_iter_history, ar_accuracy_history, marker='o', label='Autoregressive')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Evaluation Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "training_curves_compare.png"))
plt.show()
