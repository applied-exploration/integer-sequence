import torch
import torch.nn as nn
import torch.nn.functional as F

import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 0.5
EOS_token = 0
SOS_token = 1
MAX_LENGTH = 10


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, input_lang, output_lang, with_attention = False, calc_magnitude = None, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    batch_size = input_tensor.shape[1]

    decoder_input = torch.tensor([[SOS_token for _ in range(batch_size)]], device=device)

    decoder_hidden = encoder_hidden

    print("encoder hidden -> decoder hidden ", decoder_hidden.shape)

    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    use_teacher_forcing = True 

    decoder_outputs = []

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if with_attention == True:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            
            decoder_outputs.append(decoder_output)

            print("decoder output ", decoder_output.shape)
            print("decoder output ", decoder_output)
            target_tensor_unsqueezed = target_tensor[di].unsqueeze(0)
            print("target_tensor ", target_tensor_unsqueezed)
            print("target_tensor ", target_tensor_unsqueezed.shape)
            # print("target_tensor_unsqueezed.shape ", target_tensor_unsqueezed.shape)

            ''' [info] The decoder outputs now [1, 3, 16]. 3 is the batch size, 
            16 is the number of symbols we have. Each value in the matrix is a probability of that being true. 
            We either need to convert the probabilities into indexes or change the target matrix to be one-hot encoded.
            For some reason values are negative probabilities.
            '''
            loss += criterion(decoder_output, target_tensor_unsqueezed)
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if with_attention == True:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            target_tensor_unsqueezed = target_tensor[di].unsqueeze(0)
            print("target_tensor_unsqueezed.shape ", target_tensor_unsqueezed.shape)
            loss += criterion(decoder_output, target_tensor_unsqueezed)
            if decoder_input.item() == EOS_token:
                break

    if calc_magnitude is not None:
        magnitude = calc_magnitude(decoder_outputs, target_tensor, input_lang, output_lang)
        loss = loss * magnitude

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
