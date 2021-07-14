from models.rnn.rnn_plain import RNN_Plain

from lang import load_data_int_seq
from utils import accuracy_score, remove_key

import wandb
import torch
import os
# torch.backends.cudnn.enabled = False  # This is needed for Dani's local sweep


''' 0. Login to Weights and Biases '''
wsb_token = os.environ.get('WANDB_API_KEY')
if wsb_token: wandb.login(key=wsb_token)
else: wandb.login()

''' 1. Load in Data '''
output_lang, input_lang, train, X_test, y_test = load_data_int_seq()

''' 2. Define default training values '''
default_config = dict(
    symbols= "+*-0123456789t", 
    output_sequence_length= 9, 
    encoded_seq_length= 9, 
    num_epochs= 2500, 
    input_size= input_lang.n_words, 
    hidden_size= 256, 
    output_size= output_lang.n_words, 
    embedding_size= 256, 
    batch_size= 32, 
    num_gru_layers= 1,
    dropout_prob= 0.,
    calc_magnitude_on=False,
    seed=1)

training_size = 100
training_size = min(training_size, len(train))

''' 3. Initialize Weights and Biases with default values, then grab the config file (necessary for sweep) '''
wandb.init(project="integer-sequence",  config={**default_config, "training_size": training_size})
config = wandb.config

''' 4. Prepare data and config for algorithm '''
train = train[:config["training_size"]]
stripped_config = remove_key(config, "training_size")

''' 5. Create Model and train it '''
print("Experiment: Training size: {}, Batch size: {}, Epochs: {}, Dropout: {}".format(config["training_size"], config["batch_size"], config["num_epochs"], config["dropout_prob"]))
algo = RNN_Plain(**stripped_config)
algo.train(input_lang, output_lang, train)

''' 6. Calculate accuracy from the training set '''
# pred = algo.infer(input_lang, output_lang, [i[0] for i in train])
# print("Accuracy score on training set: ", accuracy_score(pred, [i[1] for i in train]))
# pred[:10]

''' 6. Calculate accuracy from the test set '''
pred = algo.infer(input_lang, output_lang, X_test[:1000])
accuracy_score(pred, y_test[:1000])
# pred[:25]


if wandb.run is not None:
    wandb.finish()
