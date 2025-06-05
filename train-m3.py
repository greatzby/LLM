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
    parser = argparse.ArgumentParser(description='NanoGPT Training with autoregressive lenient evaluation')
    parser.add_argument('--dataset', type=str, default='simple_graph', help='Name of the dataset to use')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=120, help='Size of the embeddings')
    parser.add_argument('--max_iters', type=int, default=50000, help='Total number of training iterations')
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes')
    parser.add_argument('--num_of_paths', type=int, default=20, help='Number of paths')
    parser.add_argument('--test_interval', type=int, default=100, help='Interval (in iterations) for evaluation')
    parser.add_argument('--ckpt_iter', type=int, default=10000, help='Checkpoint iteration to load (if resuming)')
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
ckpt_iter    = args.ckpt_iter
device       = args.device
temperature  = args.temperature

data_dir = os.path.join('data', f'{dataset}/{num_nodes}')
meta_path = os.path.join(data_dir, 'meta.pkl')
print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi, itos  = meta['stoi'], meta['itos']
block_size  = meta['block_size']  # 同时作为生成时的最大 token 数
top_k       = len(itos)           # 设为词表大小

# 从 meta 中获取 simple_format 标识（若不存在则默认 False）
simple_format = meta.get('simple_format', False)

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

# 注意：已移除固定随机种子，以保证每次训练均采用随机采样，从而防止模型记住固定样本顺序。
# torch.manual_seed(1337)

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

# 定义编码与解码函数（与 test 代码保持一致）
def encode(s):
    ss = s.split(" ")
    return [stoi[token] for token in ss if token in stoi]

def decode(l):
    # 与 test 代码相比，此处略有不同，但功能相当
    dec = ""
    for i in l:
        dec = dec + itos[i] + " "
    return dec[:-1]

# 定义辅助函数：根据测试文本格式（simple_format=True）时截取 prompt
def find_third_number_position(number_string):  
    numbers = number_string.split()  
    third_number_index = 2 
    position = sum(len(num) for num in numbers[:third_number_index]) + third_number_index - 1 
    return position 

# 加载图用于宽松评估
graph_path = os.path.join(data_dir, "path_graph.graphml")
if os.path.exists(graph_path):
    G = nx.read_graphml(graph_path)
else:
    print("Graph file not found, constructing a random graph for evaluation.")
    G = nx.gnp_random_graph(num_nodes, 0.3, seed=1337, directed=False)

# 定义宽松规则检查函数（与 test 代码保持一致）
def check_path(G, gen_str):
    path = re.findall(r'\d+', gen_str)
    if len(path) < 4:
        return 'wrong syntax'
    for node in path:
        if int(node) > len(itos) or int(node) < 0:
            return 'wrong syntax'
    if path[2] != path[0] or path[-1] != path[1]:
        return 'incorrect start/end'
    for i in range(2, len(path) - 1):
        if not G.has_edge(path[i], path[i + 1]):
            return f'non-existence path {(path[i], path[i + 1])}'
    return ''

@torch.no_grad()
def evaluate_autoregressive_lenient():
    """
    使用自回归生成方式对测试集（test.txt）进行评估，采样方式与 test 代码完全一致：
      1. 从测试文件中读取所有行；
      2. 根据 meta 中 simple_format 标识采用不同的 prompt 截取方式；
         - 若 simple_format 为 False，则使用 line.split(':')[0] + ':'；
         - 若 simple_format 为 True，则使用 find_third_number_position 截取 line[:pos]；
      3. 将所有 encode_texts 存入 tensor 后，随机采样 batch_size（这里为 1000）个样本，
         重复采样 10 次，调用 model.generate，并在解码后使用 .split('\n')[0] 截取第一行进行检查；
      4. 同时统计三种错误类型，返回整体准确率和错误计数。
    """
    test_file = os.path.join(data_dir, 'test.txt')
    try:
        with open(test_file, encoding='gbk') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Test file {test_file} not found. Skipping autoregressive evaluation.")
        return 0.0, {"wrong syntax": 0, "incorrect start/end": 0, "non-existence path": 0}
    
    # 根据 simple_format 处理每一行
    encode_texts = []
    ground_truth = []
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        if not simple_format:
            prompt = line.split(':')[0] + ':'
        else:
            pos = find_third_number_position(line)
            prompt = line[:pos]
        encode_texts.append(encode(prompt))
        ground_truth.append(line)
    
    # 转换为 Tensor
    encode_texts = torch.tensor(encode_texts, dtype=torch.long, device=device)
    batch_size_eval = 1000  # 与 test 代码中保持一致
    num_samples = encode_texts.shape[0]
    num_iters = 10  # 重复采样 10 次
    total_correct = 0
    total_count = 0
    error_wrong_syntax = 0
    error_incorrect_start_end = 0
    error_nonexistence = 0
    for _ in range(num_iters):
        # 随机采样
        ix = torch.randint(num_samples, (batch_size_eval,))
        x = encode_texts[ix]  # 获取采样的输入
        with torch.no_grad():
            # 调用生成函数
            y = model.generate(x, max_new_tokens=block_size, temperature=temperature, top_k=top_k)
        # 解码并取第一行（与 test 中一致）
        y_pred = [decode(y[t].tolist()).split('\n')[0] for t in range(batch_size_eval)]
        for pred in y_pred:
            symbol = check_path(G, pred)
            total_count += 1
            if symbol == "":
                total_correct += 1
            else:
                if symbol == "wrong syntax":
                    error_wrong_syntax += 1
                elif symbol == "incorrect start/end":
                    error_incorrect_start_end += 1
                elif symbol.startswith("non-existence path"):
                    error_nonexistence += 1

    accuracy = total_correct / total_count if total_count > 0 else 0.0
    error_counts = {
        "wrong syntax": error_wrong_syntax,
        "incorrect start/end": error_incorrect_start_end,
        "non-existence path": error_nonexistence
    }
    return accuracy, error_counts

# 定义记录关键层权重统计信息的函数
def record_weight_stats(model, iter_num, out_dir):
    stats = f"Iteration {iter_num}\n"
    for name, param in model.named_parameters():
        if "weight" in name and any(kw in name for kw in ["attn", "mlp", "block"]):
            tensor = param.data.cpu().numpy()
            norm_val = np.linalg.norm(tensor)
            mean_val = np.mean(tensor)
            std_val = np.std(tensor)
            stats += f"{name}: norm={norm_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}\n"
    with open(os.path.join(out_dir, "weight_stats.txt"), "a") as f:
        f.write(stats + "\n")

# ----- 模型初始化 -----
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

# 初始化 GradScaler 用于 AMP
scaler = torch.cuda.amp.GradScaler(enabled=(ptdtype == torch.float16))

# 优化器设置
weight_decay   = 1e-1
learning_rate  = 5e-4
beta1, beta2   = 0.9, 0.95
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# 学习率调度参数
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
ar_accuracy_history = []  # autoregressive 模式准确率 (宽松规则)
test_iter_history   = []

eval_interval = max_iters // 10
log_interval  = max_iters // 100
eval_iters    = max_iters // 10
eval_only     = False
always_save_checkpoint = True

t0 = time.time()
local_iter_num = 0
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
            # 记录关键层权重统计信息
            record_weight_stats(model, iter_num, out_dir)

    # 定期评估 teacher forcing 和 autoregressive 模式准确率
    if iter_num % test_interval == 0 and master_process:
        tf_acc = test_model()  # teacher forcing 模式
        ar_acc, error_counts = evaluate_autoregressive_lenient()  # autoregressive 模式 (宽松规则)
        tf_accuracy_history.append(tf_acc)
        ar_accuracy_history.append(ar_acc)
        test_iter_history.append(iter_num)
        print(f"Iteration {iter_num}: Teacher Forcing Accuracy = {tf_acc:.4f}, Autoregressive (Lenient) Accuracy = {ar_acc:.4f}")
        print(f"    Error counts - wrong syntax: {error_counts['wrong syntax']}, incorrect start/end: {error_counts['incorrect start/end']}, non-existence path: {error_counts['non-existence path']}")
        logger.info(f"Iteration {iter_num}: Teacher Forcing Accuracy = {tf_acc:.4f}, Autoregressive (Lenient) Accuracy = {ar_acc:.4f}")
        logger.info(f"    Error counts - wrong syntax: {error_counts['wrong syntax']}, incorrect start/end: {error_counts['incorrect start/end']}, non-existence path: {error_counts['non-existence path']}")

    if iter_num == 0 and eval_only:
        break

    # 前向-反向更新：使用梯度累积并结合 AMP GradScaler
    for micro_step in range(gradient_accumulation_steps):
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
plt.plot(test_iter_history, ar_accuracy_history, marker='o', label='Autoregressive (Lenient)')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Evaluation Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "training_curves_compare_Lenient.png"))
plt.show()