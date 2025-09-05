# byte pair encoding

# from importlib.metadata import version
import tiktoken

# print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace"
)

# token id
integers = tokenizer.encode(text=text, allowed_special={"<|endoftext|>"})
print(integers)

# decode to st
strings = tokenizer.decode(integers)
print(strings)

# test
test_text = "Akwirw ier"
test_integers = tokenizer.encode(text=test_text)
print(test_integers)
test_strings = tokenizer.decode(test_integers)
print(test_strings)