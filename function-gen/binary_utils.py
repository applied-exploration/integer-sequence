from typing import List

def int_to_binary_str(num: int, width: int) -> str:
    if num >= 0:
        binary = bin(num)[2:].zfill(width)
        return "0" + binary 
    else:
        binary = bin(abs(num))[2:].zfill(width)
        return "1" + binary

def binary_str_to_int(str: str, width: int) -> int:
    sign = str[0]
    num = str[1:]
    num = int(num, 2)

    if sign == "0":
        return num
    else:
        return num *- 1

bin_syms = list('01')
# char to index and index to char maps
bin_char_to_ix = { ch:i for i,ch in enumerate(bin_syms) }
bin_ix_to_char = { i:ch for i,ch in enumerate(bin_syms) }


def bin_encoder(binar: str) -> List[int]:
    # convert data from chars to indices
    output = list(binar)
    for i, ch in enumerate(binar):
        output[i] = bin_char_to_ix[ch]
    return output

def bin_decoder(encoded: List[int]) -> str:
    # convert data from chars to indices
    output = encoded.copy()
    for i, ch in enumerate(encoded):
        output[i] = bin_ix_to_char[ch]
    return ''.join(output)