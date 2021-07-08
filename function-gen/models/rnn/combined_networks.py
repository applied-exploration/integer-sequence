import torch
import torch.nn as nn
import torch.nn.functional as F

import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 0.5
EOS_token = 0
SOS_token = 1
MAX_LENGTH = 10


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, input_lang, output_lang,  with_attention = False,  calc_magnitude = None, max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    batch_size_inferred = input_tensor.shape[1]

    ''' ENCODER '''
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    encoder_hidden = encoder.initHidden()

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    
    ''' DECODER '''
    decoder_input = torch.tensor([[SOS_token for _ in range(batch_size_inferred)]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    decoder_outputs = []

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if with_attention == True:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            
            decoder_outputs.append(decoder_output)
            decoder_squeezed = decoder_output.squeeze(0)

            loss += criterion(decoder_squeezed, target_tensor[di])
            decoder_input = target_tensor[di].unsqueeze(0)  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if with_attention == True:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            ''' Here we are selecting the top prediction for each batch '''
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().view(1, -1)  # detach from history as input  

            
            decoder_output_squeezed = decoder_output.squeeze(0)
            loss += criterion(decoder_output_squeezed, target_tensor[di])


    if calc_magnitude is not None:
        magnitude = calc_magnitude(decoder_outputs, target_tensor, input_lang, output_lang)
        loss = loss * magnitude

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def infer(input_tensor, encoder, decoder, output_lang, with_attention = False):
    max_length=  10
    output_list = []

    input_length = input_tensor.size(0)
    batch_size_inferred = input_tensor.shape[1]

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    with torch.no_grad():

        ''' ENCODER '''
        encoder_hidden = encoder.initHidden(batch_size = batch_size_inferred)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        ''' DECODER '''
        decoder_input = torch.tensor([[SOS_token for _ in range(batch_size_inferred)]], device=device)
        decoder_hidden = encoder_hidden

        for di in range(max_length):
            if with_attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)  
                    
            topv, topi = decoder_output.data.topk(1)

            decoder_input = topi.squeeze().detach().view(1,-1)
            decoded_words.append(decoder_input)
            
        
        ''' PROCESS OUTPUT '''
        concatenated_output_sequences = torch.cat(decoded_words, dim=0).transpose(0,1)

        for output in concatenated_output_sequences.cpu().numpy():
            word = []
            for character in output:
                output_decoded = output_lang.index2word[character]
                word.append(output_decoded)
            
            stringified_output = ''.join(word[:-1])
            output_list.append(stringified_output)


    return output_list #decoded_words, output_sequence, decoder_attentions[:di + 1]
