from models.rnn.rnn_plain import RNN_Plain
from models.rnn.combined_networks import Loss
from lang import load_data_int_seq
from utils import accuracy_score, mae_score

import wandb

import os

WANDB_ACTIVATE = False

if WANDB_ACTIVATE:
    wsb_token = os.environ.get('WANDB_API_KEY')
    if wsb_token: wandb.login(key=wsb_token)
    else: wandb.login()

def test_algo(algo, input_lang, output_lang, train, X_test, y_test):
    algo.train(input_lang, output_lang, train)

    ''' calculate accuracy from the training set '''
    pred = algo.infer(input_lang, output_lang, [i[0] for i in train])
    print("Accuracy score on training set: ", accuracy_score(pred, [i[1] for i in train]))

    ''' calculate accuracy from the test set '''
    pred = algo.infer(input_lang, output_lang, X_test[:1000])
    print("MAE on the test set: ", mae_score(pred, y_test[:1000]))



# With Embedding layer

output_lang, input_lang, train, X_test, y_test = load_data_int_seq()

my_config ={"symbols": "+*-0123456789t", 
"output_sequence_length": 9, 
"encoded_seq_length": 9, 
"num_epochs": 2500, 
"input_size": input_lang.n_words, 
"hidden_size": 256, 
"output_size": output_lang.n_words, 
"embedding_size": 256, 
"batch_size": 32, 
"num_gru_layers": 1,
"dropout_prob": 0.,
"binary_encoding": False,
"loss":Loss.NLL,
"bidirectional": False,
"wandb_activate": WANDB_ACTIVATE}

training_size = 100
training_size = min(training_size, len(train))
train = train[:training_size]

if WANDB_ACTIVATE:
    wandb.init(project="integer-sequence",  config={**my_config, "training_size": training_size})

print("Experiment 1, with Embedding layer: Training size: {}, Batch size: {}, Epochs: {}, Dropout: {}".format(training_size, my_config["batch_size"], my_config["num_epochs"], my_config["dropout_prob"]))

algo = RNN_Plain(**my_config)

test_algo(algo, input_lang, output_lang, train, X_test, y_test)


# With Binary encoding layer

output_lang, input_lang, train, X_test, y_test = load_data_int_seq(binary_encoding= True)
training_size = 100
training_size = min(training_size, len(train))
train = train[:training_size]

my_config["binary_encoding"] = True

wandb.init(project="integer-sequence",  config={**my_config, "training_size": training_size})

print("Experiment 2, with Binary encoding: Training size: {}, Batch size: {}, Epochs: {}, Dropout: {}".format(training_size, my_config["batch_size"], my_config["num_epochs"], my_config["dropout_prob"]))

algo = RNN_Plain(**my_config)
test_algo(algo, input_lang, output_lang, train, X_test, y_test)

if WANDB_ACTIVATE:
    if wandb.run is not None:
        wandb.finish()
