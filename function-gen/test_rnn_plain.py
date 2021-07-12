from models.rnn.rnn_plain import RNN_Plain

from lang import load_data_int_seq
from utils import accuracy_score

import wandb

wandb.login()

output_lang, input_lang, train, X_test, y_test = load_data_int_seq()

my_config ={"symbols": "+*-0123456789t", 
"output_sequence_length": 9, 
"encoded_seq_length": 9, 
"num_epochs": 100, 
"input_size": input_lang.n_words, 
"hidden_size": 256, 
"output_size": output_lang.n_words, 
"embedding_size": 256, 
"batch_size": 32, 
"num_gru_layers": 3,
"dropout_prob": 0.3,
"calc_magnitude_on":False}

training_size = 100

wandb.init(project="integer-sequence",  config={**my_config, "training_size": training_size})

print("Experiment: Training size: {}, Batch size: {}, Epochs: {}, Dropout: {}".format(training_size, my_config["batch_size"], my_config["num_epochs"], my_config["dropout_prob"]))

train = train[:training_size]

algo = RNN_Plain(**my_config)
algo.train(input_lang, output_lang, train)

# calculate accuracy from the training set
pred = algo.infer(input_lang, output_lang, [i[0] for i in train])
# pred[:10]
print("Accuracy score on training set: ", accuracy_score(pred, [i[1] for i in train]))

if wandb.run is not None:
    wandb.finish()

# calculate accuracy from the test set

# pred = algo.infer(input_lang, output_lang, X_test[:1000])
# pred[:25]
# accuracy_score(pred, y_test[:1000])