SOS_token = 0
EOS_token = 1


class IntSeq:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSeq(self, seq):
        seq = [torch.Tensor(bin_encoder(int_to_binary_str(x, 6))) for x in num]


