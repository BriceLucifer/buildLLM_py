import torch

class SelfAttentionV2(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, qkv_bias: bool):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self,x:torch.Tensor):
        Keys: torch.Tensor = self.W_key(x)
        Queries: torch.Tensor = self.W_query(x)
        Values = self.W_value(x)

        AttenScores = Queries @ Keys.T
        AttenWeights = torch.softmax(
            AttenScores / Keys.shape[-1] ** 0.5, dim=-1
        )

        return AttenWeights @ Values
        