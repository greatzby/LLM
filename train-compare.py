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
    parser = argparse.ArgumentParser(description='NanoGPT Training with autoregressive evaluation')
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
dataset = args.dataset
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
max_iters = args.max_iters
num_nodes = args.num_nodes
num_of_paths = args.num_of_paths
test_interval = args.test_interval
ckpt_iter = args.ckpt_iter
device = args.device
temperature = args.temperature

data_dir = os.path.join('data', f'{dataset}/{num_nodes}')
meta_path = os.path.join(data_dir, 'meta.pkl')
print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi, itos = meta['stoi'], meta['itos']
block_size = meta['block_size']  # 同时作为生成时的最大 token 数
top_k = len(itos)  # 这里 top_k 设为词表大小，可根据任务调整
simple_format = meta.get('simple_format', False)

out_dir = f'out/{dataset}_{n_layer}_{n_head}_{n_embd}_{num_nodes}'
os.makedirs(out_dir, exist_ok=True)

# 初始化 logger
logger = get_logger(os.path.join(out_dir, "train.log"))

# 基本训练参数
gradient_accumulation_steps = 1  # 如果需要模拟大 batch，可适当增加
train_batch_size = 1024
val_batch_size = 64
batch_size = train_batch_size

# 仅支持单 GPU 训练（如需 DDP，可按实际情况扩展）
master_process = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}['bfloat16']
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 固定随机种子
torch.manual_seed(1337)

# 利用 memmap 加载数据（train 和 val）
if num_of_paths == 0:
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
else:
    train_data = np.memmap(os.path.join(data_dir, f'train_{num_of_paths}.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    """
    返回一个 mini-batch，split 取值为 "train" 或 "val"
    """
    if split == 'train':
        data = train_data
        bs = train_batch_size
    elif split == 'val':
        data = val_data
        bs = val_batch_size
    else:
        raise ValueError(f"Unknown split: {split}")
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
    使用 teacher forcing 模式在验证集上的评估，计算 token 级别准确率
    """
    X, Y = get_batch('val')
    with ctx:
        logits, _ = model(X, Y)
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == Y).float().sum().item()
    total = Y.numel()
    return correct / total

def encode(s):
    """
    将字符串根据空格分割后，通过 stoi 映射为 token 序列
    """
    ss = s.split(" ")
    # 只保留存在于词表内的 token
    return [stoi[token] for token in ss if token in stoi]

def decode(l):
    """
    将 token 序列解码为字符串（各 token 用空格分隔）
    """
    return " ".join([itos[i] for i in l])

@torch.no_grad()
def evaluate_autoregressive():
    """
    使用自回归生成方式在测试集上评估模型准确率。
    这里假设测试集文本保存在 data/<dataset>/<num_nodes>/test.txt，
    每一行作为一个样本。评估时先将文本编为 token 序列，
    然后用模型生成 max_new_tokens 个 token，再与原始 prompt 进行 token 级比较。
    """
    test_file = os.path.join(data_dir, 'test.txt')
    try:
        with open(test_file, encoding='gbk') as f:
            test_lines = [line.strip() for line in f if line.strip() != ""]
    except FileNotFoundError:
        print(f"Test file {test_file} not found. Skipping autoregressive evaluation.")
        return 0.0

    # 对每个样本进行编码
    encoded_texts = [encode(line) for line in test_lines]
    if len(encoded_texts) == 0:
        return 0.0

    # 选择最多 1000 个样本进行评估
    batch_size_eval = min(1000, len(encoded_texts))
    # 为了简单，我们对每个样本进行 padding，使得长度一致
    max_len = max(len(seq) for seq in encoded_texts[:batch_size_eval])
    padded = [seq + [0]*(max_len - len(seq)) for seq in encoded_texts[:batch_size_eval]]
    test_batch = torch.tensor(padded, dtype=torch.long, device=device)

    # 使用模型的 generate() 方法自回归生成
    generated = model.generate(test_batch, max_new_tokens=block_size, temperature=temperature, top_k=top_k)
    total_tokens = 0
    correct_tokens = 0
    for i in range(batch_size_eval):
        orig_len = len(encoded_texts[i])
        # 生成的序列中，前 orig_len 个 token对应输入 prompt，后面的为生成部分
        gen_tokens = generated[i].tolist()[orig_len:]
        # 这里简单比较生成token与 prompt 内 token 对应位置的匹配情况
        N = min(orig_len, len(gen_tokens))
        total_tokens += N
        for j in range(N):
            if test_batch[i, j].item() == gen_tokens[j]:
                correct_tokens += 1
    ar_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    return ar_accuracy

# ----- 模型初始化：加载 checkpoint -----
if num_of_paths == 0:
    ckpt_path = os.path.join(out_dir, f'{ckpt_iter}_ckpt.pt')
else:
    ckpt_path = os.path.join(out_dir, f'{ckpt_iter}_ckpt_{num_of_paths}.pt')
print(f"Loading checkpoint from {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 配置优化器（如果需要继续训练）
optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=5e-4, betas=(0.9, 0.95), device_type=device_type)

# ----- 开始训练循环 -----
iter_num = 0
best_val_loss = 1e9

# 用于记录训练动态
train_loss_history = []
train_iter_history = []
tf_accuracy_history = []   # teacher forcing（验证集）准确率
ar_accuracy_history = []   # autoregressive（测试集）准确率
test_iter_history = []

t0 = time.time()
local_iter_num = 0
raw_model = model  # 如无 DDP，则直接使用 model
running_mfu = -1.0

while iter_num <= max_iters:
    # ---------- 学习率调度（cosine warmup decay） ----------
    warmup_iters = max_iters // 20
    lr_decay_iters = max_iters
    min_lr = 5e-4 / 10
    if iter_num < warmup_iters:
        lr = 5e-4 * iter_num / warmup_iters
    elif iter_num > lr_decay_iters:
        lr = min_lr
    else:
        decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        lr = min_lr + coeff * (5e-4 - min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # ---------- 定期评估 loss 并保存 checkpoint ----------
    if iter_num % (max_iters // 10) == 0 and master_process:
        losses = {}
        for split in ['train', 'val']:
            loss_vals = []
            # 这里采样一定迭代数以获得较稳定的 loss 估计
            for _ in range(max_iters // 10):
                X, Y = get_batch(split)
                with ctx:
                    _, loss = model(X, Y)
                loss_vals.append(loss.item())
            losses[split] = np.mean(loss_vals)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            checkpoint_dict = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
            }
            ckpt_save_path = os.path.join(out_dir, f'{iter_num}_ckpt.pt') if num_of_paths == 0 else os.path.join(out_dir, f'{iter_num}_ckpt_{num_of_paths}.pt')
            torch.save(checkpoint_dict, ckpt_save_path)
            print(f"Checkpoint saved to {ckpt_save_path}")
            logger.info(f"Checkpoint saved to {ckpt_save_path}")

    # ---------- 定期评估验证集和测试集准确率 ----------
    if iter_num % test_interval == 0 and master_process:
        tf_acc = test_model()  # teacher forcing 模式在验证集上的准确率
        ar_acc = evaluate_autoregressive()  # autoregressive 模式在测试集上的准确率
        tf_accuracy_history.append(tf_acc)
        ar_accuracy_history.append(ar_acc)
        test_iter_history.append(iter_num)
        print(f"Iteration {iter_num}: Teacher Forcing Accuracy = {tf_acc:.4f}, Autoregressive Accuracy = {ar_acc:.4f}")
        logger.info(f"Iteration {iter_num}: Teacher Forcing Accuracy = {tf_acc:.4f}, Autoregressive Accuracy = {ar_acc:.4f}")

    # ---------- 前向-反向更新：训练一步 ----------
    X, Y = get_batch('train')
    optimizer.zero_grad(set_to_none=True)
    with ctx:
        logits, loss = model(X, Y)
    # 对 loss 进行反向传播并梯度裁剪
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # ---------- 日志记录 ----------
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    lossf = loss.item()
    train_loss_history.append(lossf)
    train_iter_history.append(iter_num)
    if local_iter_num >= 5:
        mfu = raw_model.estimate_mfu(batch_size, dt)
        running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
    print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, MFU {running_mfu*100:.2f}%")
    logger.info(f"iter {iter_num}: loss {lossf:.4f}")
    iter_num += 1
    local_iter_num += 1

# ---------- 绘制训练曲线并保存图片 ----------
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
plt.savefig(os.path.join(out_dir, "training_curves.png"))
plt.show()