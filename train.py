import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, casual_mask
from model import build_transformer

from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import warnings
from pathlib import Path

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_targt, max_len, device):
    sos_idx = tokenizer_targt.token_to_id('[SOS]')
    eos_idx = tokenizer_targt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every token we get from the deocder
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target (decoder input)
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the decoder
        decoder_output = model.decoder(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token
        prob = model.project(decoder_output[:, -1])

        # Select the token with the max probability (because it is a greedy function)
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)

def run_validation(model, validation_data, tokenizer_src, tokenizer_targt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    # Size of the control window (just us a default value)
    console_width = 80

    with torch.no_grad():
        for batch in validation_data:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_targt, max_len, device)
            
            source_text = batch['src_txt'][0]
            target_text = batch['targt_txt'][0]
            model_out_text = tokenizer_targt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # print to the console
            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break
    # if writer:
        # TorchMetrics CharErrorRate, BLEU, WordErrorRate

def get_all_sentences(data, lang):
    for item in data:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, data, lang):

    # config['tokenizer_file'] = '../tokenizers/tokenizer-{0}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(data, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    data_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_targt"]}', split="train")

    #Build Tokenizer
    tokenizer_src = get_or_build_tokenizer(config, data_raw, config['lang_src'])
    tokenizer_targt = get_or_build_tokenizer(config, data_raw, config['lang_targt'])

    # keep 90% from training and 10% for validation
    train_data_size = int(0.9 * len(data_raw))
    val_data_size = len(data_raw) - train_data_size
    train_data_raw, val_data_raw = random_split(data_raw, [train_data_size, val_data_size])

    train_ds = BilingualDataset(train_data_raw, tokenizer_src, tokenizer_targt, config['lang_src'], config['lang_targt'], config['seq_len'])
    val_ds = BilingualDataset(val_data_raw, tokenizer_src, tokenizer_targt, config['lang_src'], config['lang_targt'], config['seq_len'])

    max_len_src = 0
    max_len_targt = 0

    for item in data_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        targt_ids = tokenizer_src.encode(item['translation'][config['lang_targt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_targt = max(max_len_targt, len(targt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_targt}')

    train_data_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_data_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_data_loader, val_data_loader, tokenizer_src, tokenizer_targt


def get_model(config, vocab_src_len, vocab_targt_len):
    model = build_transformer(vocab_src_len, vocab_targt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    # Define the device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_data_loader, val_data_loader, tokenizer_src, tokenizer_targt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_targt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        
        batch_iterator = tqdm(train_data_loader, desc=f'Processing epoch {epoch:02d}')
        
        for batch in batch_iterator:
            model.train()

            encoder_input = batch['encoder_input'].to(device) # size => (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # size => (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # size => (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # size => (batch, 1, seq_len, seq_len)

            # Run the tensors through the transfomer
            encoder_output = model.encode(encoder_input, encoder_mask) # size => (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # size => (batch, seq_len, d_model)
            project_output = model.project(decoder_output) # size => (batch, seq_len, targt_vocab_size)

            label = batch['label'].to(device) # size => (batch, seq_len)

            # (batch, seq_len, targt_vocab_size) => (batch * seq_len, targt_vocab-size)
            loss = loss_fn(project_output.view(-1, tokenizer_targt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropogate the loss
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        
        run_validation(model, val_data_loader, tokenizer_src, tokenizer_targt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        

        # save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    # warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)      



            

