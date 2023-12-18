import torch
import sentencepiece as spm

def create_tokenizer(model='unigram_4096_simplified.model'):
    sp = spm.SentencePieceProcessor()
    sp.load('../tokenizers/' + model)
    
    encode = lambda text: torch.tensor(sp.EncodeAsIds(text), dtype=torch.long)
    decode = lambda tokens: sp.DecodeIds(tokens)
    
    return encode, decode, sp
