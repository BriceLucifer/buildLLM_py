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
# q & k
queries : torch.Tensor = sa_v2.W_query(inputs)
keys : torch.Tensor = sa_v2.W_key(inputs)

atten_scores = queries @ keys.T
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

