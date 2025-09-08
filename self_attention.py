import torch

'''
    L1 Normalize: atten_score / sum of atten_score
    - 保证所有分量之和 = 1
    - 结果可以解释为比例或概率分布（前提：分量非负）
'''
def l1_normalize(atten_score: torch.Tensor) -> torch.Tensor:
    return atten_score / atten_score.sum()

'''
    L2 Normalize: atten_score / ||atten_score||
    - 保证向量的模长 (L2 norm) = 1
    - 保留方向信息，去掉大小信息
    - 常用于余弦相似度计算
'''
def l2_normalize(atten_score: torch.Tensor) -> torch.Tensor:
    return atten_score / torch.norm(atten_score, p=2)


'''
    Softmax Normalize: e^(atten_score) / sum of e^(atten_score)
'''
def self_made_softmax_normalize(atten_score: torch.Tensor) -> torch.Tensor:
    return torch.exp(atten_score) / torch.exp(atten_score).sum(dim=0)

# input token embedding (simple example)
inputs = torch.tensor([
 [0.43, 0.15, 0.89], # Your (x^1)
 [0.55, 0.87, 0.66], # journey (x^2)
 [0.57, 0.85, 0.64], # starts (x^3)
 [0.22, 0.58, 0.33], # with (x^4)
 [0.77, 0.25, 0.10], # one (x^5)
 [0.05, 0.80, 0.55] # step (x^6)
])

query = inputs[1] # journey
attn_score_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_score_2[i] = torch.dot(x_i, query)

print(attn_score_2)
# dot 点乘 如果方向距离相互接近,weights比较大; 如果距离大,weights 比较小
'''
    dot(a,b) = |a| * |b| * cos(a,b)
'''

# 归一化操作 主要是获得总和为1的注意力权重 单位向量 其实就是更加关注方向 而不是大小了
attn_weights_2_tmp = l1_normalize(atten_score=attn_score_2)
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# self made softmax 归一化操作
# warning: may have value stable question
attn_weights_2_naive = self_made_softmax_normalize(atten_score=attn_score_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# 使用torch自带的softmax 归一化操作
attn_weight_2 = torch.softmax(attn_score_2, dim=0)
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weight_2[i] * x_i
print(context_vec_2)


# 接下来开始计算所有的注意力权重
attn_scores = torch.empty(6,6)
# for 循环效率慢
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i,j] = torch.dot(x_i, x_j)
attn_scores = inputs @ inputs.T
print(attn_scores)

# 归一化
attn_weights = torch.softmax(attn_scores, dim=1) # 每一行进行归一化
print(attn_weights)

# 计算上下文向量
all_content_vecs = attn_weights @ inputs
print(all_content_vecs)