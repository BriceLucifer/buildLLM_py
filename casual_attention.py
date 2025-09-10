import torch
from self_attentionV2 import SelfAttentionV2

# we use a simple tensor
inputs = torch.tensor([
 [0.43, 0.15, 0.89], # Your (x^1)
 [0.55, 0.87, 0.66], # journey (x^2)
 [0.57, 0.85, 0.64], # starts (x^3)
 [0.22, 0.58, 0.33], # with (x^4)
 [0.77, 0.25, 0.10], # one (x^5)
 [0.05, 0.80, 0.55] # step (x^6)
])

# input dimension and output dimension
d_in = inputs.shape[1] # d_in = x_2.shape[0] 一样的
d_out = 2
torch.manual_seed(789)

# self attention v2
sa_v2 = SelfAttentionV2(d_in=d_in, d_out=d_out,qkv_bias=False)

# q & k & v
queries : torch.Tensor = sa_v2.W_query(inputs)
keys : torch.Tensor = sa_v2.W_key(inputs)
values : torch.Tensor = sa_v2.W_value(inputs)

# attention scores
atten_scores = queries @ keys.T
# normal
atten_weights = torch.softmax(atten_scores / keys.shape[-1] ** 0.5, dim=-1)
print(atten_weights)

# 上下文长度
context_length = atten_weights.shape[0]
# 设置掩码为1 长度为上下文长度 对角线为1
# mask_simple = torch.tril(torch.ones(context_length, context_length))
# masked_simple = mask_simple * atten_weights
# print(masked_simple)

# # 归一化
# row_sums = masked_simple.sum(dim=-1, keepdim=True)
# masked_simple_norm = masked_simple / row_sums
# print(masked_simple_norm)
# 被注释的代码虽然能够完成工作 但是不足够更高的效率去计算

# 多了一步掩码分析
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
'''
    [0,1,1,1]
    [0,0,1,1]
    [0,0,0,1]
    [0,0,0,0]
'''
masked = atten_scores.masked_fill(mask=mask.bool(), value=-torch.inf)
'''
mask.bool():
    [false, true, true, true]
    [false, false, true, true]
    [false, false, false, true]
    [false, false, false, false]

replace true with -inf or false with the true value
'''
print(masked)

# normalization
atten_weights = torch.softmax(masked / keys.shape[-1] ** 0.5 , dim=-1)
print(atten_weights) # same result with previous operation with more efficient computation

# content vec 上下文向量
context_vec = atten_weights @ values
print(values)

# 3.5.2 dropout
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6,6)
'''
    [1,1,1,1,1,1]
    [1,1,1,1,1,1]
    [1,1,1,1,1,1]
    [1,1,1,1,1,1]
    [1,1,1,1,1,1]
    [1,1,1,1,1,1]
'''
print(dropout(example))
'''
    当使用dropout 50% 减少依赖的时候 其他的元素会以 1/0.5 的权重放大
    [2., 2., 0., 2., 2., 0.],
    [0., 0., 0., 2., 0., 2.],
    [2., 2., 2., 2., 0., 2.],
    [0., 2., 2., 0., 0., 2.],
    [0., 2., 0., 2., 0., 2.],
    [0., 2., 2., 2., 2., 0.]]
'''

# 来看看atten_weight
print(dropout(atten_weights))

# 自己创建一个因果注意力类
class CausalAttention(torch.nn.Module):
    '''
        __init__()
        d_in: 输入维度
        d_out: 输出维度
        context_length: 上下文长度 
        dropout: dropout 丢弃率
        qkv_bias: 是否矫正
    '''
    def __init__(self, d_in:int, d_out:int, context_length:int, dropout:float, qkv_bias: bool = False) -> None:
        super().__init__()
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer (
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch size, token numbers, dimension in
        b, token_nums, d_int = x.shape
        queries = self.W_query(x)
        keys: torch.Tensor = self.W_key(x)
        values = self.W_value(x)

        atten_scores:torch.Tensor = queries @ keys.transpose(1,2) # shape[1]和shape[2]交换
        atten_scores.masked_fill( self.mask.bool()[:token_nums, :token_nums], -torch.inf) # pyright: ignore[reportCallIssue]
        atten_weights = torch.softmax(atten_scores/keys.shape[-1] ** 0.5, dim=-1)

        context_vec = self.dropout(atten_weights) @ values
        return context_vec


# test CasualAttention
torch.manual_seed(123)
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)
context_length = batch.shape[1]
ca = CausalAttention(
    d_in=d_in, 
    d_out=d_out, 
    context_length=context_length, 
    dropout=0.5, 
    qkv_bias=False
)

context_vec : torch.Tensor = ca(batch)
print("Context Vec Shape: ", context_vec.shape)