import re
from SimpleTokenizerV2 import SimpleTokenizerV2
from importlib.metadata import version
import tiktoken
from GPTDatasetV1 import GPTDatasetV1
import torch

tokenizer = tiktoken.get_encoding("gpt2")

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:] #remove first 50 tokens

context_size = 4         #1
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")
#prints "sliding window"

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

#THIS CREATES PYTORCH DATASET CLASS OF INPUT -> TARGETS
#Need to implement efficient data loader that iterates over input dataset and returns pytorch tensors
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
        tokenizer = tiktoken.get_encoding("gpt2")                         #1
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)   #2
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,     #3
            num_workers=num_workers     #4
        )

        return dataloader


dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)      #1
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)

