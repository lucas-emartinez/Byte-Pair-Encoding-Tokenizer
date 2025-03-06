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