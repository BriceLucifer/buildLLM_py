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

if __name__ == "__main__":
    main()