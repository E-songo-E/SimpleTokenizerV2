import re
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import os

with open('cool-textz-eng.txt', 'r') as file:
    RAW_TEXT = file.read()
reprocessed = re.findall(r'\b\w+\b', RAW_TEXT)
preprocessed = [item.strip() for item in reprocessed if item.strip()]
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=50000, special_tokens=["<|unk|>", "<|pad|>", "<|bos|>", "<|eos|>"])
tokenizer.train_from_iterator(preprocessed, trainer=trainer)
tokenizer.save("bpe_tokenizer.json")
vocab = tokenizer.get_vocab()
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")
print(f"First 10 tokens in vocabulary: {list(vocab.keys())[:10]}")
token_to_id = vocab
id_to_token = {id_: token for token, id_ in vocab.items()}
class SimpleTokenizerV2:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text):
        encoded = self.tokenizer.encode(text.lower())
        return encoded.ids 

    def decode(self, ids):
        decoded = self.tokenizer.decode(ids)
        return decoded
tokenizer_v1 = SimpleTokenizerV2(tokenizer)
sample_text = "the rottweiler fell through the horizon."
encoded_v1 = tokenizer_v1.encode(sample_text)
decoded_v1 = tokenizer_v1.decode(encoded_v1)
print("Tokenizer V1 (With BPE):")
print(f"Encoded: {encoded_v1}")
print(f"Decoded: {decoded_v1}")
