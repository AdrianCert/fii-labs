import re
import nltk
from collections import defaultdict, Counter
from pathlib import Path


class NgramLanguageModel:
    def __init__(self, n_gram=2):
        self.n_gram = n_gram
        self.ngram_freqs = None
        self.unigram_freqs = None
        self.vocabulary_size = None

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = nltk.word_tokenize(text)
        return tokens

    def build_ngram_model(self, corpus):
        if isinstance(corpus, str):
            tokens = self.preprocess_text(corpus)
        else:
            tokens = corpus

        self.unigram_freqs = Counter(tokens)
        
        ngrams = list(nltk.ngrams(tokens, self.n_gram, pad_right=True, pad_left=True, 
                                  left_pad_symbol="<s>", right_pad_symbol="</s>"))
        self.ngram_freqs = Counter(ngrams)
        
        self.vocabulary_size = len(set(tokens))

    def laplace_smoothing(self):
        smoothed_model = defaultdict(float)
        for ngram in self.ngram_freqs:
            smoothed_model[ngram] = (self.ngram_freqs[ngram] + 1) / (self.unigram_freqs[ngram[:-1]] + self.vocabulary_size)
        return smoothed_model

    def calculate_sentence_probability(self, sentence):
        tokens = self.preprocess_text(sentence)
        ngrams = list(nltk.ngrams(tokens, self.n_gram, pad_right=True, pad_left=True, 
                                  left_pad_symbol="<s>", right_pad_symbol="</s>"))
        # breakpoint()
        smoothed_model = self.laplace_smoothing()

        probability = 1.0
        for ngram in ngrams:
            probability *= smoothed_model.get(ngram, 1 / (self.unigram_freqs[ngram[:-1]] + self.vocabulary_size))
        
        return probability

    def get_ngram_frequencies(self):
        return self.ngram_freqs

    def get_unigram_frequencies(self):
        return self.unigram_freqs


if __name__ == "__main__":
    corpus = Path(__file__).parent.joinpath("corpus.txt").read_text()

    bigram_model = NgramLanguageModel(n_gram=2)
    bigram_model.build_ngram_model(corpus)
    test_sentence = "Conversaţia a mai durat mult"
    test_sentence = "Ziua e frumoasa"
    test_sentence = "Vreau sa merg la masa"
    test_sentence = "Îmi amintesc foarte vag"
    sentence_probability = bigram_model.calculate_sentence_probability(test_sentence)

    print(f"Probability of the sentence '{test_sentence}': {sentence_probability}")

    # print(bigram_model.get_ngram_frequencies())
    # print(bigram_model.get_unigram_frequencies())