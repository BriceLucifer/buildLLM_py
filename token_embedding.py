import torch


def main():
    input_ids = torch.tensor([2,3,5,2])
    # 词汇表大小
    vocab_size = 6
    # 输出维度
    output_dims = 3

    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dims)
    print(embedding_layer.weight)
    print(embedding_layer(input_ids))
    # no mater where the token Id is, the same token Id is gonna reflexed to the same vector positon

    '''two method for positon information embedding
    1. absolute positional embedding (gpt choice)
        - add an unique positon embedding (same dimention of the original token)
    2. relative postional embedding
        - focus on the distance between one token to another
    '''


if __name__ == "__main__":
    main()