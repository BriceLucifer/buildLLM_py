# GPT dict
GPT_CONFIG_124M = {
    "vocab_size": 50257,            # 词汇表大小
    "context_length": 1024,         # 上下文长度
    "emb_dim": 768,                  # 嵌入维度
    "n_heads": 12,                  # 注意力头的数量
    "n_layers": 12,                 # 层数
    "drop_rate": 0.1,               # dropout rate
    "qkv_bias": False               # 查询-键-值偏置
}

import torch
import torch.nn as nn

# 简单站位置 transformer block
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

# 站位 归一化层 NormLayer
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, epa:float = 1e-5) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class DummyGPTModel(nn.Module):
    def __init__(self, cfg:dict) :
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    
    def forward(self, in_idx: torch.Tensor) -> torch.Tensor :
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
# tokenizer

import tiktoken

# init tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

# init a DummyGPTModel
torch.manual_seed(123)
model:nn.Module = DummyGPTModel(GPT_CONFIG_124M)
logits: torch.Tensor = model(batch)
print("output shape:", logits.shape)
print(logits)

# example for norm layer
torch.manual_seed(123)
batch_example = torch.randn(2,5)
print(batch_example)
layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
out: torch.Tensor = layer(batch_example)
print(out)

# 归一化之前 检查均值和方差
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:", mean)
print("Variance:", var)

'''
    我们常说的“标准化（standardization）”就是：
    减去均值：让数据的中心在 0
    除以标准差：让数据的离散程度（spread）统一为 1
    这样处理后，不同样本的数据都映射到 零中心、单位尺度 的标准空间。
'''

# 归一化
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:", out_norm)
print("Mean:", mean)
print("Variance:", var)

# 提高可读性 关掉科学计数法
torch.set_printoptions(sci_mode=False)
print("Mean:", mean)
print("Variance:", var)

# 整理为一个class
class LayerNorm(nn.Module):
    def __init__(self, emb_dim:int) -> None:
        super().__init__()
        self.eps = 1e-5             # eps epsilon 常数 归一化过程的时候会加在方差上防止除0错误
        self.scale = nn.Parameter(torch.ones(emb_dim))      # scale 可缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))     # shift 偏移参数 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# example of LayerNorm
ln = LayerNorm(emb_dim=5)
out_ln: torch.Tensor = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True, unbiased=False)
print("Mean:", mean)
print("Var:", var)

# GELU 函数
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        return 0.5 * x * (
            1 + torch.tanh(
                (torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )

# 直观对比
import matplotlib.pyplot as plt

gelu, relu = GELU(), nn.ReLU()
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8,3))

for i, (y, labal) in enumerate(zip([y_gelu, y_relu], ["GELU", "RELU"]), 1):
    plt.subplot(1,2,i)
    plt.plot(x,y)
    plt.title(f"{labal} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{labal}(x)")
    plt.grid(True)

plt.tight_layout()
# plt.show()

class FeedForward(nn.Module):
    def __init__(self, cfg:dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # dim_out = dim_in * 4
            GELU(),                                         # dim_out = dim_in
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])   # dim_out = dim_in / 4
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        return self.layers(x)

# example 
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes: list, use_shortcut: bool):
        super().__init__()
        self.use_shortcut = use_shortcut
        # 五层网络
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()), 
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        for layer in self.layers:
            layer_out: torch.Tensor = layer(x)
            if self.use_shortcut and x.shape == layer_out.shape :
                x = x + layer_out
            else:
                x = layer_out
        return x

# example of normal neural network and shortcut neural network
layer_size = [3,3,3,3,3,1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes=layer_size,
    use_shortcut=False
)

# define a backward function for compute gradients
def print_gradients(model:nn.Module, x:torch.Tensor):
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss_r: torch.Tensor = loss(output, target)

    loss_r.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}") # pyright: ignore[reportOptionalMemberAccess]

# normal neural network
print_gradients(model_without_shortcut, sample_input)
print("\n")
# shortcut neural network
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes=layer_size,
    use_shortcut=True
)
print_gradients(model_with_shortcut, sample_input)

# connect the attention layer and linear layer
from multihead_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg=cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        # 在注意力块添加快捷链接 后续用
        shortcut: torch.Tensor = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 再次建立快捷链接
        shortcut = x 
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

# example
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(cfg=GPT_CONFIG_124M)
output: torch.Tensor = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

# 实现GPT架构
class GPTModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            * [TransformerBlock(cfg=cfg) for _ in range(cfg["n_layers"])]
        )
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# example of use GPTModel
torch.manual_seed(123)
model:nn.Module = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:", batch, "\n")
print("Output batch:", out.shape, "\n")
print(out)

# 计算参数
total_params = sum( p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
# weight tying 权重共享
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)
# 所以需要减去输出层的参数量
total_params_gpt2 = (
    total_params - sum( p.numel() for p in model.out_head.parameters())
)

print(f"Number of trainable parameters " 
      f"considering weight tying: {total_params_gpt2:,}"
)
# 计算对象中1.63亿参数的内存需求
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")

# the method for gpt2 to generate text
def generate_text_simpel(model:nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int) -> torch.Tensor:
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

# example 尝试用Hello, I am 上下文作为模型输入来调用generate_text_simple
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) # 增加batch维度
print("encoded_tensor.shape:", encoded_tensor.shape)
# 然后模型设置为.eval() 禁用dropout等只在训练期间使用的随机组件
model.eval()
out = generate_text_simpel(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
print("Output length:", len(out[0]))
# 使用分词器 把id 转换为文本
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)