import re
from SimpleTokenizerV1 import SimpleTokenizerV1
from SimpleTokenizerV2 import SimpleTokenizerV2

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text) #splits raw_text into tokens, organized by r''
preprocessed = [item.strip() for item in preprocessed if item.strip()] #removes excess whitespace?

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

#print(len(vocab.items()))
#for i, item in enumerate(list(vocab.items())[-5:]):
    #print(item)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

#note: UNK REPRESENTS AN UNKOWN
tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))