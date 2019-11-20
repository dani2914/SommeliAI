import numpy as np


class Lda:
    def __init__(self, num_topics: int, num_terms: int, num_docs: int) -> None:
        self.num_topics = num_topics
        self.num_terms = num_terms
        self.num_docs = num_docs
        self.alpha = 1.0
        self.eta = 1.0
        self.var_gamma = np.ones((num_docs, num_topics)) / num_topics
        self.var_lambda = np.ones((num_topics, num_terms))
        self.log_prob_w = np.ones((num_topics, num_terms))
        self.vocab = {}


def load_model(model_root: str):
    # model_root is a string
    with open(model_root + '.other', 'r') as other_file:
        parser = other_file.readline().split()
        num_topics = int(parser[1])
        parser = other_file.readline().split()
        num_terms = int(parser[1])
        parser = other_file.readline().split()
        num_docs = int(parser[1])
        parser = other_file.readline().split()
        alpha = float(parser[1])
        parser = other_file.readline().split()
        eta = float(parser[1])
        parser = other_file.readline().split()
        vocab = parser[1]

    model = Lda(num_topics, num_terms, num_docs)
    model.alpha = alpha
    model.eta = eta

    with open(vocab, "r") as file:
        words = file.read().splitlines()

    model.vocab = dict(zip(range(len(words)), words))

    return model


def print_topics(model: Lda):
    for k in range(0, model.num_topics):
        lambdak = list(model.var_lambda[k, :])
        lambdak = lambdak / sum(lambdak)
        temp = zip(lambdak, range(0, len(lambdak)))
        temp = sorted(temp, key=lambda x: x[0], reverse=True)
        for i in range(0, 10):
            print(f"{model.vocab[temp[i][1]]}", end=" ")
        print()
