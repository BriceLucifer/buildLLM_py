# 实现多头注意力机制
import torch

from casual_attention import CausalAttention


# 简单wrapper
class MulitHeadAttentionWrapper(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)


# simple usage

torch.manual_seed(123)

# we use a simple tensor
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your (x^1)
        [0.55, 0.87, 0.66],  # journey (x^2)
        [0.57, 0.85, 0.64],  # starts (x^3)
        [0.22, 0.58, 0.33],  # with (x^4)
        [0.77, 0.25, 0.10],  # one (x^5)
        [0.05, 0.80, 0.55],  # step (x^6)
    ]
)

# input dimension and output dimension
d_in = inputs.shape[1]  # d_in = x_2.shape[0] 一样的
print(d_in)
d_out = 1
batch = torch.stack((inputs, inputs), dim=0)

context_length = batch.shape[1]
mha = MulitHeadAttentionWrapper(
    d_in=d_in, d_out=d_out, context_length=context_length, dropout=0.0, num_heads=2
)

context_vec: torch.Tensor = mha(batch)

print(context_vec)
print("context vec shape:", context_vec.shape)


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = torch.nn.Linear(d_in, d_out, qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 讲解数学原理之后
        b, num_token, d_in = x.shape
        keys: torch.Tensor = self.W_key(x)
        queries: torch.Tensor = self.W_query(x)
        values: torch.Tensor = self.W_value(x)

        keys = keys.view(b, num_token, self.num_heads, self.head_dim)
        queries = queries.view(b, num_token, self.num_heads, self.head_dim)
        values = values.view(b, num_token, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_token, :num_token]  # pyright: ignore[reportCallIssue]

        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        attn_weights: torch.Tensor = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_token, self.d_out)

        context_vec = self.out_proj(context_vec)
        return context_vec


torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
