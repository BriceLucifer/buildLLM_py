import torch

class SelfAttention_v1(torch.nn.Module):
    def __init__(self, d_in:int, d_out:int):
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        Query = x @ self.W_query
        Key = x @ self.W_key
        Value = x @ self.W_value

        Scores = Query @ Key.T # Omega
        Attention_weight = torch.softmax(
            Scores / Key.shape[-1] ** 0.5, dim=-1
        )

        ContextVec = Attention_weight @ Value
        return ContextVec