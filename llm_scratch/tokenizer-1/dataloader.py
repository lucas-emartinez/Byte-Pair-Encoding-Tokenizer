import re
from tokenizerv1 import SimpleTokenizerV1
from tokenizerv2 import SimpleTokenizerV2

def dataloader():
    '''
    This function loads the raw text from the file.
    '''
    try:
        with open("the-verdict.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
            return raw_text
        #print("Total number of characters: ", len(raw_text))
        #print(raw_text[:99])
    except FileNotFoundError:
        print("Error: El archivo 'the-verdict.txt' no se encontró.")
    except Exception as e:
        print(f"Se produjo un error: {e}")


def split_text(raw_text: str) -> list[str]:
     '''
     This function tokenizes the raw text.
     It splits the text by commas, periods, spaces, and special characters.
     It also removes empty strings.
     '''
     result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
     return result

def vocabulary(preprocessed: list[str]) -> list[str]:
    '''
    This function creates a vocabulary from the preprocessed text.
    It returns a dictionary with the token as the key and the integer as the value.
    '''
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    return vocab

raw_text = dataloader()
preprocessed = split_text(raw_text)
preprocessed = [item for item in preprocessed if item.strip()]
vocab = vocabulary(preprocessed)

tokenizer2 = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

ids = tokenizer2.encode(text)
print(ids)

decoded = tokenizer2.decode(ids)
print(decoded)

