from models.rnn.rnn_plain import RNN_Plain

from lang import load_data_int_seq
from utils import accuracy_score, remove_key

import wandb
import torch
import os
# torch.backends.cudnn.enabled = False  # This is needed for Dani's local sweep


wsb_token = os.environ.get('WANDB_API_KEY')
if wsb_token: wandb.login(key=wsb_token)
else: wandb.login()


output_lang, input_lang, train, X_test, y_test = load_data_int_seq()

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
    calc_magnitude_on=False)

training_size = 100
training_size = min(training_size, len(train))

wandb.init(project="integer-sequence",  config={**default_config, "training_size": training_size})
config = wandb.config

print("Experiment: Training size: {}, Batch size: {}, Epochs: {}, Dropout: {}".format(training_size, config["batch_size"], config["num_epochs"], config["dropout_prob"]))

train = train[:config["training_size"]]

stripped_config = remove_key(config, "training_size")

algo = RNN_Plain(**stripped_config)
algo.train(input_lang, output_lang, train)

''' calculate accuracy from the training set '''
# pred = algo.infer(input_lang, output_lang, [i[0] for i in train])
# print("Accuracy score on training set: ", accuracy_score(pred, [i[1] for i in train]))
# pred[:10]

''' calculate accuracy from the test set '''
pred = algo.infer(input_lang, output_lang, X_test[:1000])
accuracy_score(pred, y_test[:1000])
# pred[:25]


if wandb.run is not None:
    wandb.finish()
