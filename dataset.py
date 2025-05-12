import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, data, tokenizer_src, tokenizer_targt, src_lang, targt_lang, seq_len) -> None:
        super().__init__()
        self.data = data
        self.tokenizer_src = tokenizer_src
        self.tokenizer_targt = tokenizer_targt
        self.src_lang = src_lang
        self.targt_lang = targt_lang
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor([tokenizer_targt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_targt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_targt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        src_target_pair = self.data[index]
        src_txt = src_target_pair['translation'][self.src_lang]
        targt_txt = src_target_pair['translation'][self.targt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_txt).ids
        dec_input_tokens = self.tokenizer_targt.encode(targt_txt).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        # Add SOS and EOS to the sourch text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add SOS to the decoder
        decoder_input = torch.cat(
            [
               self.sos_token,
               torch.tensor(dec_input_tokens, dtype=torch.int64),
               torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64) 
            ]
        )

        # Add EOS to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64) 
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # size => (seq_len)
            "decoder_input": decoder_input, # size => (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # size => (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), # size => (1, seq_len) & (1, seq_len, seq_len)
            "label": label, # size => (seq_len)
            "src_txt": src_txt,
            "targt_txt": targt_txt
        }
    
def casual_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0