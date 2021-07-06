from models.rnn.rnn_plain import RNN_Plain
from models.rnn.rnn_attention import RNN_Attention

from lang import load_data_int_seq
from utils import accuracy_score

output_lang, input_lang, train, X_test, y_test = load_data_int_seq()

algo = RNN_Plain(symbols = "+*-0123456789t", 
output_sequence_length = 9, 
encoded_seq_length = 9, 
num_epochs = 5000, 
input_size = input_lang.n_words, 
hidden_size = 256, 
output_size=output_lang.n_words, 
embedding_size = 28, 
batch_size = 32, 
calc_magnitude_on=False)

algo.train(input_lang, output_lang, train)

pred = algo.infer(input_lang, output_lang, X_test[:1000])
pred[:25]

accuracy_score(pred, y_test[:1000])