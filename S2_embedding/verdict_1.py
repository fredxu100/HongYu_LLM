import re
from SimpleTokenizerV1 import SimpleTokenizerV1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
#print("Total number of character:", len(raw_text))
#print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text) #splits raw_text into tokens, organized by r''
preprocessed = [item.strip() for item in preprocessed if item.strip()] #removes excess whitespace?
#print(len(preprocessed)) #prints number of tokens
#print(preprocessed[:30]) #prints first 30 elements

all_words = sorted(set(preprocessed)) #converts preprocessed into a set, removing duplicates, and orders by ascii
vocab_size = len(all_words) #prints the length of the set
#print(vocab_size) 

vocab = {token:integer for integer,token in enumerate(all_words)} #??? converts into dict? token: ID
#SAME FUNCTIONALITY AS THE FOLLOWING: 
#all_words = ['!', 'Gisburn', 'Jack', 'a']
#vocab = {}
#integer = 0

#for token in all_words:
#    vocab[token] = integer
#    integer = integer + 1


for i, item in enumerate(vocab.items()): #prints
    print(item)
    if i >= 50:
        break

tokenizer = SimpleTokenizerV1(vocab) #passes in dictionary
text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)

#print(ids)
#print(tokenizer.decode(ids))


