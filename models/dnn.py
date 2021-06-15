import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_size, output_size, seed = 5, hidden_layer_param = []):
        """Initialize parameters and build model.
        Params
        ======
            input_size (int): Dimension of each input sequence
            output_size (int): Dimension of each output sequence
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DNN, self).__init__()
        self.seed = torch.manual_seed(seed)

        new_hidden_layer_param = hidden_layer_param.copy()

        # --- Input layer --- #
        self.fc_in = nn.Linear(input_size, new_hidden_layer_param[0])

        # --- Hidden layers --- #
        if len(new_hidden_layer_param) < 2: self.hidden_layers = []
        else: self.hidden_layers = nn.ModuleList([nn.Linear(new_hidden_layer_param[i], new_hidden_layer_param[i+1]) for i in range(len(new_hidden_layer_param)-1)])

        # --- Output layer --- #
        self.fc_out = nn.Linear(new_hidden_layer_param[-1], output_size)
        

    def forward(self, state):
        """Build a network that maps input -> output values."""

        x = F.relu(self.fc_in(state))

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        x = self.fc_out(x)

        
        x1 = torch.exp(F.log_softmax(x[:9]))
        x2 = torch.exp(F.log_softmax(x[9:18]))
        x3 = torch.exp(F.log_softmax(x[18:27]))
        x4 = torch.exp(F.log_softmax(x[27:36]))
        x5 = torch.exp(F.log_softmax(x[36:45]))
        x6 = torch.exp(F.log_softmax(x[45:54]))
        x7 = torch.exp(F.log_softmax(x[54:63]))
        x8 = torch.exp(F.log_softmax(x[63:72]))
        x9 = torch.exp(F.log_softmax(x[72:81]))

        # x = x.view(9, -1)
        # one_hot = torch.Tensor([F.log_softmax(y) for y in x])

        print(x1)
        print(x2)

        return x1,x2,x3,x4,x5,x6,x7,x8,x9

        # return self.fc3(x)
