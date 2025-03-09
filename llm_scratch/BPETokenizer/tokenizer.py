from collections import Counter

class BytePairTokenizer:
    def __init__(self, vocab_size=276):
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def get_stats(self, ids):
        """Find the most frequent pair of tokens."""
        return Counter(zip(ids, ids[1:]))

    def merge(self, ids, top_pair, idx):
        """Merge the most frequent pair into a new token."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == top_pair[0] and ids[i + 1] == top_pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def fit(self, text):
        """Train the tokenizer on the provided text."""
        tokens = list(text.encode("utf-8"))
        ids = list(tokens)  # Copy to avoid modifying the original list

        for i in range(self.num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"Merging {top_pair} into a new token {idx}")
            ids = self.merge(ids, top_pair, idx)
            self.merges[top_pair] = idx

        # Update vocab with merged tokens
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def decode(self, ids):
        """Decode a list of token IDs back to a string."""
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def encode(self, text: str):
        """Encode a string into a list of token IDs."""
        tokens = list(text.encode("utf-8"))
        while True:
            stats = self.get_stats(tokens)
            if not stats:
                break  # Nothing else to merge
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break  # Nothing else to merge
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

# Ejemplo de uso
tokenizer = BytePairTokenizer()
tokenizer.fit("aavdbbaa")  # Entrenar el tokenizador
encoded = tokenizer.encode("aavdbbaa")  # Codificar texto
decoded = tokenizer.decode(encoded)  # Decodificar de vuelta

print("Encoded:", encoded)
print("Decoded:", decoded)

import regex as re

gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
print(re.findall(gpt2pat, "What's the weather in Tokyo?"))


import tiktoken
# GPT-2 (does not merge spaces)
enc = tiktoken.get_encoding("gpt2")
print(enc.encode("Hello world"))

enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode("Hello world"))

enc = tiktoken.get_encoding("p50k_base")
print(enc.encode("Hello world"))


import sentencepiece as spm

with open("toy.txt", "w", encoding="utf-8") as f:
  f.write("SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training. SentencePiece implements subword units (e.g., byte-pair-encoding (BPE) [Sennrich et al.]) and unigram language model [Kudo.]) with the extension of direct training from raw sentences. SentencePiece allows us to make a purely end-to-end system that does not depend on language-specific pre/postprocessing.")

options = dict(
    input="toy.txt",
    input_format="text",
    # output spec
    model_prefix="tok400" # ew, turn off normalization
)

# train a sentencepiece model on it
# the settings here are (best effort) those used for training Llama 2
import os

options = dict(
  # input spec
  input="toy.txt",
  input_format="text",
  # output spec
  model_prefix="tok400", # output filename prefix
  # algorithm spec
  # BPE alg
  model_type="bpe",
  vocab_size=400,
  # normalization
  normalization_rule_name="identity", # ew, turn off normalization
  remove_extra_whitespaces=False,
  input_sentence_size=200000000, # max number of training sentences
  max_sentence_length=4192, # max number of bytes per sentence
  seed_sentencepiece_size=1000000,
  shuffle_input_sentence=True,
  # rare word treatment
  character_coverage=0.99995,
  byte_fallback=True,
  # merge rules
  split_digits=True,
  split_by_unicode_script=True,
  split_by_whitespace=True,
  split_by_number=True,
  max_sentencepiece_length=16,
  add_dummy_prefix=True,
  allow_whitespace_only_pieces=True,
  # special tokens
  unk_id=0, # the UNK token MUST exist
  bos_id=1, # the others are optional, set to -1 to turn off
  eos_id=2,
  pad_id=-1,
  # systems
  num_threads=os.cpu_count(), # use ~all system resources
)

spm.SentencePieceTrainer.train(**options)

sp = spm.SentecePieceProcessor()
sp.load('tok400.model')
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]




