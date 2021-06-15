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

        return self.fc_out(x)

        # return self.fc3(x)
