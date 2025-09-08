import torch
from data_loader import create_dataloader_v1

def main():
    # input_ids = torch.tensor([2,3,5,2])
    # 词汇表大小
    vocab_size = 50257
    # 输出维度
    output_dims = 256

    # read raw text
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    # 使用上一节的dataloader
    max_length = 4
    data_loader = create_dataloader_v1(txt=raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)

    data_iter = iter(data_loader)
    inputs, targets = next(data_iter)
    # 一次8行， 4列
    print("Token IDs:\n", inputs)
    print("\nInputs shapes:", inputs.shape)

    # torch.manual_seed(123)
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dims)
    token_embeddings = token_embedding_layer(inputs)
    print("Token embeddings shape:",token_embeddings.shape)
    # no mater where the token Id is, the same token Id is gonna reflexed to the same vector positon

    # use absolute postional embedding so we need to create a same dimension embedding layer
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dims)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length)) # arange from 0..4
    print(pos_embeddings.shape)

    '''two method for positon information embedding
    1. absolute positional embedding (gpt choice)
        - add an unique positon embedding (same dimention of the original token)

        token embedding     : [token one](256 dim) [token two](256 dim) ...
                                    +                   +
        postion embedding   : [pos  one](256 dim) [pos two](256 dim)   ...
                                    =                   =
        input embedding

    2. relative postional embedding
        - focus on the distance between one token to another
    '''

    input_embeddins = token_embeddings + pos_embeddings
    print(input_embeddins.shape)

if __name__ == "__main__":
    main()