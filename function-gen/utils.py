from typing import List

syms = list('+*-0123456789t')
# char to index and index to char maps
char_to_ix = { ch:i for i,ch in enumerate(syms) }
ix_to_char = { i:ch for i,ch in enumerate(syms) }

def eq_encoder(eq: str) -> List[int]:
    # convert data from chars to indices
    output = list(eq)
    for i, ch in enumerate(eq):
        output[i] = char_to_ix[ch]
    return output

def eq_decoder(encoded: List[int]) -> str:
    # convert data from chars to indices
    output = encoded.copy()
    for i, ch in enumerate(encoded):
        output[i] = ix_to_char[ch]
    return output

# eq_decoder(eq_encoder('1*23t+23'))
