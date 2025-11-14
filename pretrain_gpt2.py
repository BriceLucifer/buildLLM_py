import matplotlib.pyplot as plt
import tiktoken
import torch
import torch.nn as nn
from matplotlib.ticker import MaxNLocator

from data_loader import create_dataloader_v1

# import the gpt core model from chapter4
from gpt_core import GPTModel, generate_text_simple

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  # we set the context to 256 in order to train in a regular laptop
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,  # can set to 0
    "qkv_bias": False,
}

# init the modal
torch.manual_seed(123)
# model
model = GPTModel(cfg=GPT_CONFIG_124M)
model.eval()


# function for text to token id
# encoding
def text_to_token_ids(text: str, tokenizer: tiktoken.core.Encoding) -> torch.Tensor:
    encoded = tokenizer.encode(text=text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).squeeze()
    return encoded_tensor


# decoding
def token_ids_to_text(
    token_ids: torch.Tensor, tokenizer: tiktoken.core.Encoding
) -> str:
    flat = token_ids.squeeze(0)
    return tokenizer.decode(tokens=flat.tolist())


start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)

print("Output text:\n", token_ids_to_text(token_ids=token_ids, tokenizer=tokenizer))

# next step we compute the loss of the text generator
# example
inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
targets = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])

# 提供给logits向量 其实就是被求exp的原始变量，翻译不好
with torch.no_grad():
    # 屏蔽grad 因为没有开始training
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)
print(probas, probas.shape)

# 接下来通过argmax函数用于概率
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)
# 目标targets
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
# actual value
print(f"Outputs batch 1: {token_ids_to_text(token_ids.flatten(), tokenizer)}")

# original softmax
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

# 求log
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
# make more 第3节 第四节
print(log_probas)

# 求mean
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

# 求non_avg_log_probas
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)

# the steps to calculate the loss
"""
    1. logits
    2. probas   <- softmax(logits)
    3. target_probas <-
    4. log_probas <- log(target_probas)
    5. mean_probas <- mean(log_probas)
    6. non_neg_probas <- -mean_probas
"""
# just use cross entropy
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets_flats", targets_flat.shape)
# use cross_entropy in pytorch
loss = nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)

# perplexity(困惑度): 表示模型预测文本序列中下一个词的不确定性
"""
    perplexity = exp(loss)
"""

# read the text
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

# calculate the len of the words
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

# try train on verdict.txt
max_length = 6  # 6 token each for fast training
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# use a dataloader
torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)
val_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0,
)

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)


# define a cross entropy function
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(device)
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)


# additional function
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # 禁用dropout 目的产出稳定可以复现的结果
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
    model.train()

    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", ""))
    model.train()


# the main training model function
def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epoches,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epoches):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                # 评测model
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}",
                    f"Val loss {val_loss:.3f}",
                )

        # 简单打印和生成看看效果
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
# 我增加了epoch
num_epoches = 20
# 先不训练
# train_losses, val_losses, tokens_seen = train_model_simple(
#     model,
#     train_loader,
#     val_loader,
#     optimizer,
#     device,
#     num_epoches=num_epoches,
#     eval_freq=5,
#     eval_iter=1,
#     start_context="Every effort moves you",
#     tokenizer=tokenizer,
# )


def plot_losses(epoches_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epoches_seen, train_losses, label="Training loss")
    ax1.plot(epoches_seen, val_losses, linestyle="-", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()


# epochs_tensor = torch.linspace(0, num_epoches, len(train_losses))
# plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

model.to("cpu")
model.eval()

# tokenizer = tiktoken.get_encoding("gpt2")
# token_ids = generate_text_simple(
#     model=model,
#     idx=text_to_token_ids("Every effort moves you know", tokenizer),
#     max_new_tokens=25,
#     context_size=GPT_CONFIG_124M["context_length"],
# )
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


vocab: dict[str, int] = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}

inverse_vocab: dict[int, str] = {v: k for k, v in vocab.items()}
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()

torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])


def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab.get(next_token_id)}")


print_sampled_tokens(probas=probas)


def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


temperature = [1, 0.1, 5]
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperature]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperature):
    rects = ax.bar(
        x + i * bar_width, scaled_probas[i], bar_width, label=f"Temperature = {T}"
    )
ax.set_ylabel("Probability")
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()
