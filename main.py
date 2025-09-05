import re
import urllib.request
from tokenizer import SimpleTokenizerV1

# download data
def download_nova() -> str:
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)
    return file_path

def easy_tokenizer(text: str) -> list[str]:
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',text) 
    result = [item for item in preprocessed if item.strip()]
    return result

def token_id(tokens: list[str])->list[str]:
    all_words = sorted(set(tokens))
    return all_words


def main():
    # download file
    file_path :str = download_nova()

    # open_file
    with open(file_path,'r', encoding="utf-8") as f:
        raw_text = f.read()
    
    # print("Total number of character:", len(raw_text))
    # print(raw_text[:99])
    vr = easy_tokenizer(raw_text)

    all_words = token_id(vr)
    vocab_size = len(all_words)
    print(vocab_size)

    vocab: dict[str,int]= { token:integer for integer,token in enumerate(all_words) }

    tokenizer = SimpleTokenizerV1(vocab)
    text = """"It's the last he painted, you know,"
           Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    # 词元id
    print(ids)
    print(tokenizer.decode(ids))
    # for i,item in enumerate(vocab.items()):
    #     print(item)
    #     if i >= 50:
    #         break
    

    # print(vr[:30])
    # print(len(vr[:30]))

    # text: str = "Hello, world. This, is a test."
    # print(text)
    # # result: list[str] = re.split(r'(\s)', text)
    # result: list[str] = re.split(r'([,.]|\s)', text)
    # result = [item for item in result if item.strip()]
    # print(result)

    # text = "Hello, world. Is this-- a test?"
    # print(text)
    # result = easy_tokenizer(text=text)
    # print(result)

if __name__ == "__main__":
    main()
