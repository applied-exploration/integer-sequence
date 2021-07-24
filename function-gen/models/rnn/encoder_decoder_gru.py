import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from typing import List, Tuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

BINARY_NUM = 32

class EncoderRNN(nn.Module):

    def __init__(self, input_size:int, hidden_size:int, embedding_size:int, batch_size:int, cnn_output_depth: List[int] = [], cnn_kernel_size:int = 3, cnn_batch_norm:bool=True, cnn_activation:bool=True, num_gru_layers:int = 1, dropout:float = 0.0,  seed:int = 1, bidirectional:bool=False, binary_encoding:bool = False) -> None:

        super(EncoderRNN, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_gru_layers = num_gru_layers
        self.cnn_output_depth = cnn_output_depth
        self.bidirectional = bidirectional
        self.binary_encoding = binary_encoding
        self.embedding = nn.Embedding(input_size, embedding_size)
    
        if binary_encoding == True:
            embedding_size = BINARY_NUM

        gru_input = embedding_size
        
        self.cnn = []

        if len(cnn_output_depth) > 0:
            cnn_output_depth.insert(0, embedding_size)
            cnn_list = []
            for i in range(len(cnn_output_depth)-1):
                cnn_list.append(nn.Conv1d(cnn_output_depth[i], cnn_output_depth[i+1], kernel_size=cnn_kernel_size, stride=1, padding=1))
                if cnn_batch_norm: cnn_list.append(nn.BatchNorm1d(cnn_output_depth[i+1]))
                if cnn_activation: cnn_list.append(nn.ReLU())

            self.cnn = nn.ModuleList(cnn_list)
            gru_input = cnn_output_depth[-1]

        self.gru = nn.GRU(gru_input, hidden_size, num_layers=num_gru_layers, dropout = dropout, bidirectional=self.bidirectional)
        


    def forward(self, input, hidden):
        ''' 
            Embedding layer: 
                Size:  [ dictionary size, length of embedded vector ]
                Input: Receives just encoded categories, eg.: [2, 3, 5]
            GRU layer:
                Size: [ length of embedded vector, hidden_size ]
                Input: [ sequence length, batch_size, embedding size ]​	
                    eg.: seq_len: 1, batch_size: 2, embedding size: 3
                          [ [ [ 0.5, 0.2, 0.3 ],
                              [ 0.2, 0.6, 0.7 ] ] ]  
        
            Additional Comment:
                 We need to unsqueeze the embedding output, 
                because it gives out [num_batch, symbol_encoded_to_vector]
                and GRU needs [sequence_len, num_batch, symbol_encoded_to_vector]
                unsqueeze effectively adds a single dimensional array at the location specified, squeeze takes away 1-s

        '''
        # print("===> Encoder Input")
        # print("input ", input.shape)
        # print("hidden ", hidden.shape)
        # print("<================= ")

        
        
        if self.binary_encoding == True:
            embedded = dec2bin(input, BINARY_NUM)
        else:
            embedded = self.embedding(input)


        if len(self.cnn)>0:
            embedded = embedded.transpose(0,1).transpose(1,2)

            output = embedded
            for cnn_layer in self.cnn:
                output = cnn_layer(output)
                print(output.shape)

            output = output.transpose(0,1).transpose(0,2)
        else:        
            output = embedded
        
        output, hidden = self.gru(output, hidden) # output [seq_len, batch size, hid dim * num directions] | hidden [n layers * num directions, batch size, hid dim]
        
        # print("===")
        
        if self.bidirectional: 
            hidden_forward = hidden[-2,:,:]
            hidden_backward = hidden[-1,:,:]
            hidden = torch.cat((hidden_forward, hidden_backward), dim = 1)
        else:
            hidden = hidden.squeeze(0)


        # print("===> Encoder Output")
        # print("output " , output.shape)
        # print("hidden ", hidden.shape )
        # print("<=================")
        return output, hidden

    def initHidden(self, batch_size = None):
        if batch_size == None : batch_size = self.batch_size

        if self.bidirectional: return torch.zeros(2 * self.num_gru_layers, batch_size, self.hidden_size, device=device)
        else: return torch.zeros(self.num_gru_layers, batch_size, self.hidden_size, device=device) 


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, embedding_size: int, batch_size: int, num_gru_layers: int = 1, dropout: float = 0.0, seed:int = 1, bidirectional_encoder:bool=False) -> None:
        super(DecoderRNN, self).__init__()
        
        self.seed = torch.manual_seed(seed)

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_gru_layers

        self.bidirectional_encoder = bidirectional_encoder

        self.embedding = nn.Embedding(output_size, embedding_size)
        
        bidirectional_multiplier = 1
        if self.bidirectional_encoder: 
            bidirectional_multiplier =2

        self.gru = nn.GRU(embedding_size, hidden_size * bidirectional_multiplier, dropout = dropout, num_layers= num_gru_layers)
        
        self.out = nn.Linear(hidden_size * bidirectional_multiplier, output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input, hidden):
        ''' 
            Embedding layer: 
                Size:  [ dictionary size, length of embedded vector ]
                Input: Receives just encoded categories, eg.: [2, 3, 5]
            GRU layer:
                Size: [ length of embedded vector, hidden_size ]
                Input: [ sequence length, batch_size, embedding size ]​	
                    eg.: seq_len: 1, batch_size: 2, embedding size: 3
                          [ [ [ 0.5, 0.2, 0.3 ],
                              [ 0.2, 0.6, 0.7 ] ] ]  
        '''
        # print("===> Decoder Input")
        # print("input ", input.shape)
        # print("hidden ", hidden.shape)
        # print("<================= ")

        embedding = self.embedding(input)
        
        output = embedding 
        output = F.relu(output)

        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
   
        output = output.unsqueeze(0)
       
        # print("===> Decoder Output")
        # print("output " , output.shape)
        # print("hidden ", hidden.shape )
        # print("<=================")

        return output, hidden



MAX_LENGTH = 10

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=device)
