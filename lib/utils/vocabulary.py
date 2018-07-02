class Vocabulary(object):
    """A vocabulary wrapper, contains a word_to_index dictionary and a index_to_word list"""

    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = []
        self.index = 0

    def add_word(self, word):
        if not word in self.word_to_index:
            self.word_to_index[word] = self.index
            self.index_to_word.append(word)
            self.index += 1

    def __call__(self, word):
        if type(word) == int:
            return self.index_to_word[word]
        else:
            if not word in self.word_to_index:
                return self.word_to_index[unicode('<UNK>')]
            return self.word_to_index[word]

    def __len__(self):
        return self.index

"""Builds a Vocabulary object"""
def build_vocab(sentences, min_occurrences=5, progress_bar=True):
    from collections import Counter
    from tqdm import tqdm
    counter = Counter()
    for sentence in (tqdm(sentences) if progress_bar else sentences):
        counter.update(sentence)

    # a word must appear at least min_occurrence times to be included in the vocabulary
    words = [word for word, count in counter.items() if count >= min_occurrences]

    # Creating a vocabulary object
    vocab = Vocabulary()
    vocab.add_word(unicode("<SOS>"))
    vocab.add_word(unicode("<EOS>"))
    vocab.add_word(unicode("<UNK>"))

    # Adds the words from the captions to the vocabulary
    for word in words:
        vocab.add_word(word)
    return vocab

def save_vocab(vocab, data_dir, min_occurrences=5):
    import pickle

    save_file = "{}/data/datasets/vocab_{}.pkl".format(data_dir, min_occurrences)
    with open(save_file, "wb") as f:
        pickle.dump(vocab, f)
    return save_file

def load_vocab(data_dir, min_occurrences=5):
    import pickle

    save_file = "{}/data/datasets/vocab_{}.pkl".format(data_dir, min_occurrences)
    with open(save_file, "rb") as f:
        return pickle.load(f)
