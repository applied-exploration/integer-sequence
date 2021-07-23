import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, embedding_size:int, batch_size:int, num_gru_layers:int = 1, dropout:float = 0.0, seed:int = 1, bidirectional:bool=False) -> None:
        super(EncoderRNN, self).__init__()
        
        self.seed = torch.manual_seed(seed)

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_gru_layers
        self.bidirectional = bidirectional
     
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=num_gru_layers, dropout = dropout, bidirectional=self.bidirectional)



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
        
        
        embedded = self.embedding(input) 
        output = embedded
        output, hidden = self.gru(output, hidden) # output [seq_len, batch size, hid dim * num directions] | hidden [n layers * num directions, batch size, hid dim]
        
        if self.bidirectional: hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        # print("===> Encoder Output")
        # print("output " , output.shape)
        # print("hidden ", hidden.shape )
        # print("<=================")
        return output, hidden

    def initHidden(self, batch_size = None):
        if batch_size == None : batch_size = self.batch_size

        if self.bidirectional: return torch.zeros(2 * self.num_layers, batch_size, self.hidden_size, device=device)
        else: return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device) 


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
    
    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def initHidden(self, encoder_hidden):
        # if batch_size == None : batch_size = self.batch_size

        # if self.bidirectional_encoder: return torch.zeros(2 * self.num_layers, batch_size, self.hidden_size, device=device)
        # else: return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device) 
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden



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