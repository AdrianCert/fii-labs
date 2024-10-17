import re
from collections import defaultdict, Counter


class BPEParser:
    def __init__(self, vocab_size):
        """
        Initialize the BPEParser with the desired vocabulary size.
        """
        self.vocab_size = vocab_size
        self.vocab = {}  # To store token pairs and their counts
        self.bpe_codes = {}  # To store the final BPE merge operations

    def get_vocab(self, corpus):
        """
        Create the initial vocabulary from the corpus. Split each word into characters with spaces.
        """
        vocab = defaultdict(int)
        for word in corpus:
            word = " ".join(list(word)) + " </w>"  # Add </w> to mark end of word
            vocab[word] += 1
        return vocab

    def get_stats(self, vocab):
        """
        Count frequency of all symbol pairs in the vocabulary.
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        """
        Merge the most frequent pair in the vocabulary.
        """
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        new_vocab = {}
        for word in vocab:
            # Replace the bigram with the merged symbol
            new_word = pattern.sub("".join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def train(self, corpus):
        """
        Train the BPE on a given corpus. Build the vocabulary until we reach the desired size.
        """
        vocab = self.get_vocab(corpus)
        print(f"Initial vocab: {vocab}")

        for i in range(self.vocab_size):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            # Find the most frequent pair
            best_pair = max(pairs, key=pairs.get)
            print(f"Step {i + 1}: Most frequent pair: {best_pair}")

            # Merge the best pair in the vocabulary
            vocab = self.merge_vocab(best_pair, vocab)
            self.bpe_codes[best_pair] = i  # Store the merge operation

        self.vocab = vocab  # Final vocabulary after training
        print(f"Final vocab: {self.vocab}")
        print(f"BPE codes: {self.bpe_codes}")

    def encode(self, word):
        """
        Encode a word using the learned BPE codes.
        """
        word = " ".join(list(word)) + " </w>"  # Split the word into characters
        symbols = word.split()

        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            best_pair = None
            for pair in pairs:
                if pair in self.bpe_codes:
                    best_pair = pair
                    break
            if best_pair is None:
                break
            # Merge the best pair
            new_word = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    new_word.append("".join(best_pair))
                    i += 2
                else:
                    new_word.append(symbols[i])
                    i += 1
            symbols = new_word

        return symbols


# Example usage:

# Sample corpus
corpus = ["low", "lower", "lowest", "newest"]

# Initialize BPEParser with desired vocabulary size (e.g., 10 merges)
bpe_parser = BPEParser(vocab_size=10)

# Train BPE on the corpus
bpe_parser.train(corpus)

# Encode a word using the trained BPE
encoded_word = bpe_parser.encode("lowest")
print(f"Encoded 'lowest': {encoded_word}")
