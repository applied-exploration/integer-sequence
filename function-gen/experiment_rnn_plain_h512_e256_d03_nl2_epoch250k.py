from models.rnn.rnn_plain import RNN_Plain

from lang import load_data_int_seq
from utils import accuracy_score

output_lang, input_lang, train, X_test, y_test = load_data_int_seq()
print("Experiment 1: Training size: 10k, Batch size: 32, Hidden: 512, Embedding: 256, Dropout: 0.3, No of hidden layers: 2, Epochs: 250k, ")
train = train[:10000]

algo = RNN_Plain(symbols = "+*-0123456789t", 
output_sequence_length = 9, 
encoded_seq_length = 9, 
num_epochs = 250000, 
input_size = input_lang.n_words, 
hidden_size = 512, 
output_size=output_lang.n_words, 
embedding_size = 256, 
batch_size = 32, 
num_gru_layers = 2,
dropout_prob = 0.3,
calc_magnitude_on=False)

algo.train(input_lang, output_lang, train)

# calculate accuracy from the training set

pred = algo.infer(input_lang, output_lang, [i[0] for i in train])
# pred[:10]
print("Accuracy score on training set: ", accuracy_score(pred, [i[1] for i in train]))

# calculate accuracy from the test set

pred = algo.infer(input_lang, output_lang, X_test)
# pred[:25]
print("Accuracy score on test set: ", accuracy_score(pred, y_test))
