import re

class SimpleTokenizerV1:
    '''
    Tokenizer V1:
    - Simple tokenizer that uses a dictionary to map strings to integers and vice versa.
    - It preprocesses the text by splitting it into tokens and removing whitespace.
    - It encodes the text into a list of integers and decodes it back to text.
    '''
    
    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join(self.int_to_str[id] for id in ids)
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text