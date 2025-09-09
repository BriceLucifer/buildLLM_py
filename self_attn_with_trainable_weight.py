'''
    scaled dot-product attention
'''

import torch

# we use a simple tensor
inputs = torch.tensor([
 [0.43, 0.15, 0.89], # Your (x^1)
 [0.55, 0.87, 0.66], # journey (x^2)
 [0.57, 0.85, 0.64], # starts (x^3)
 [0.22, 0.58, 0.33], # with (x^4)
 [0.77, 0.25, 0.10], # one (x^5)
 [0.05, 0.80, 0.55] # step (x^6)
])

# w_q w_k w_v
x_2 = inputs[1]
d_in = inputs.shape[1] # d_in = x_2.shape[0] 一样的
d_out = 2

# init the w_q, w_k, w_v
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# compute the query_vec key_vec value_vec
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

# we need all the keys and values of all inputs
keys = inputs @ W_key
values = inputs @ W_value
print("Keys shape:", keys.shape)
print("Values shape:", values.shape)

# compute attention score
keys_2 = keys[1]
atten_score_2 = query_2 @ keys.T
print(atten_score_2)

# softmax normalization
d_k = keys.shape[-1]
atten_weights_2 = torch.softmax(atten_score_2 / d_k ** 0.5, dim=-1)

# compute value
context_vec_2 = atten_weights_2 @ values
print("Context vec2:",context_vec_2)

# use the class SelfAttention v1
from self_attentionV1 import SelfAttention_v1

torch.manual_seed(123)
self_attention = SelfAttention_v1(d_in=d_in, d_out=d_out)
print("Self Made class attention: ")
print(self_attention(inputs))

# use the class SelfAttention v2
from self_attentionV2 import SelfAttentionV2

torch.manual_seed(123)
self_attention = SelfAttentionV2(d_in=d_in, d_out=d_out, qkv_bias=False)
print("Self v2:")
print(self_attention(inputs))


# Test 
v1 = SelfAttention_v1(d_in=d_in, d_out=d_out)
v2 = SelfAttentionV2(d_in=d_in, d_out=d_out, qkv_bias=False)
v1.W_key.data = v2.W_key.weight.data.T.clone()
v1.W_query.data = v2.W_query.weight.data.T.clone()
v1.W_value.data = v2.W_value.weight.data.T.clone()

print(v1(inputs))
print(v2(inputs))