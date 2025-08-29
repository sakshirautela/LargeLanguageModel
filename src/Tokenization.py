import tiktoken  # Byte Pair Encoding
import torch
import re
from torch.utils.data import Dataset, DataLoader


class TokenGen:
    def __init__(self):
        self.encodings = {
            "gpt2": tiktoken.get_encoding("gpt2"),
            "gpt3": tiktoken.get_encoding("p50k_base"),  # Commonly associated with GPT-3 models
            "gpt4": tiktoken.get_encoding("cl100k_base")  # Used for GPT-4 and later versions
        }

        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids):
        return self.tokenizer.decode(ids)


# class SimpleTokenV1: # learning
#     def __init__(self, text):
#          tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
#         tokens = [t.strip() for t in tokens if t.strip()]
#         self.str_to_int = {t: i for i, t in enumerate(sorted(set(tokens)))}
#         self.int_to_str = {i: t for t, i in self.str_to_int.items()}
#
#     def encode(self, text):
#         tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
#         tokens = [t.strip() for t in tokens if t.strip()]
#         return [self.str_to_int[t] for t in tokens]
#
#     def decode(self, ids):
#         text = " ".join([self.int_to_str[i] for i in ids])
#         text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
#         return text


# tokenizers=TokenGen()
# text = (
#     "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
#     "of someunknownPlace. me"
# )
# integers = tokenizers.encode(text)
# print(integers)
# print(tokenizers.decode(integers))
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        if len(token_ids) <= max_length:
            self.input_ids.append(torch.tensor(token_ids[:-1]))
            self.target_ids.append(torch.tensor(token_ids[1:]))
        else:
            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i:i + max_length]
                target_chunk = token_ids[i + 1:i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = TokenGen()
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


def createEmbwdding(self, raw_text):
    dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=2, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)
    vocab_size = 50257
    output_dim = 256

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)