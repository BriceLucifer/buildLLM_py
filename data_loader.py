import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    """
        __init__() txt:文字 tokenizer:分词器 max_langth:长度, stride:步伐
    """
    def __init__(self, txt, tokenizer, max_length, stride) -> None:
        self.input_ids = [] 
        self.target_id = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_id.append(torch.tensor(target_chunk))
    """
        __len__ 长度 返回数据集长度
    """
    def __len__(self):
        return len(self.input_ids)

    """
        __getitem 返回指定行
    """
    def __getitem__(self, idx) :
        return self.input_ids[idx],  self.target_id[idx]
    
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # init tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    # init dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    # create the dataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

def main():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    # 创建数据加载器
    dataloader = create_dataloader_v1(txt=raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    # 数据迭代器 input 和 target
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

if __name__ == "__main__":
    main()