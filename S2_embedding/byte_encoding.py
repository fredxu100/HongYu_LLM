from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken")) #version 0.14.0

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace." #someunkownPlace recognized by bypte pairing because it creates tokens
     #does NOT use |unk| tokens
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

test_input = ("Akwirw ier")
test_int = tokenizer.encode(test_input)
print(test_int)