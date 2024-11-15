from pathlib import Path
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import matplotlib.pyplot as plt
import numpy as np
import typing
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

ASSETS_DIR = Path(__file__, "../.assets")
GLOVE_DIR = ASSETS_DIR / "glove"
WO2VE_DIR = ASSETS_DIR / "words2vec"


class Visualize:
    @staticmethod
    def view_2d(words, matrix, title):
        plt.figure(figsize=(10, 10))
        for i, word in enumerate(words):
            x, y, *_ = matrix[i]
            plt.scatter(x, y)
            plt.annotate(word, (x, y), fontsize=12)
        plt.title(title)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show(block=False)


class BaseModel:
    model = None
    name: str = "Model"

    def load(): ...

    def ensure_model(self):
        if not self.model:
            self.load()

    def get_model(self):
        self.ensure_model()
        return self.model

    def cosine_similarity(self, word1, word2):
        self.ensure_model()
        vec1, vec2 = self.model[word1], self.model[word2]
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def visualize(self, words, ensure_loading: bool = True, method: str = "TSNE"):
        if ensure_loading:
            self.ensure_model()
        word_vectors = np.array([self.model[word] for word in words])

        if method == "TSNE":
            tsne_model = TSNE(n_components=2, perplexity=3, random_state=42)
            word_vectors_2d = tsne_model.fit_transform(word_vectors)

            Visualize.view_2d(
                words=words,
                matrix=word_vectors_2d,
                title=f"{self.name}: Word Embedding Visualization with TNSE",
            )
        elif method == "PSE":
            pass
        else:
            Visualize.view_2d(
                words=words,
                matrix=word_vectors,
                title=f"{self.name}: Word Embedding Visualization",
            )


class GloveModel(BaseModel):
    name = "Glove"

    def __init__(self, dim: int):
        super().__init__()
        self.glove_path = Path(GLOVE_DIR, f"glove.6B.{dim}d.txt")
        self.dim = dim
        if not self.glove_path.exists():
            raise ValueError(
                f"the file glove.6B.{dim}d.txt not available in {GLOVE_DIR}"
            )
        self.word2vec_path = Path(GLOVE_DIR, f"glove.6B.{self.dim}d.word2vec.txt")

    def load(
        self,
    ):
        if not self.word2vec_path.exists():
            glove2word2vec(self.glove_path.as_posix(), self.word2vec_path.as_posix())
        self.model = KeyedVectors.load_word2vec_format(
            self.word2vec_path.as_posix(),
            binary=False,
        )
        return self.model


class GoogleWo2VecModel(BaseModel):
    name = "Words2Vec"

    def __init__(self, dim: int):
        super().__init__()
        self.source_path = Path(WO2VE_DIR, "GoogleNews-vectors-negative300.bin")
        self.dim = 300
        self.model = None

    def load(
        self,
    ):
        self.model = KeyedVectors.load_word2vec_format(
            self.source_path.as_posix(),
            binary=True,
        )
        return self.model


if __name__ == "__main__":
    # part I
    input_words = [
        # "frogs", "toad", "litoria", "leptodactylidae", "rana",
        "cat",
        "drive",
        "driver",
        "car",
        "wheel",
        "king",
        "queen",
        "doctor",
        "engineer",
        "apple",
        "banana",
        "river",
        "mountain",
        "love",
        "happiness",
    ]

    glove_model = GloveModel(100)
    wo2ve_model = GoogleWo2VecModel(300)

    models: typing.List[BaseModel] = [glove_model, wo2ve_model]

    for model in models:
        model.visualize(input_words)
        model.visualize(input_words, method="..")

    input("Press any key to continue ...")

if __name__ == "__main__":
    # part II
    related_pairs = [("king", "queen"), ("doctor", "nurse"), ("apple", "banana")]
    unrelated_pairs = [("king", "apple"), ("mountain", "happiness")]
    wo2ve_model = GoogleWo2VecModel(300)

    print("Similarity for related pairs:")
    for words_pair in related_pairs:
        similarity = wo2ve_model.cosine_similarity(*words_pair)
        print("{} - {}: {:.4f}".format(*words_pair, similarity))

    print("Similarity for unrelated pairs:")
    for words_pair in unrelated_pairs:
        similarity = wo2ve_model.cosine_similarity(*words_pair)
        print("{} - {}: {:.4f}".format(*words_pair, similarity))

    input("Press any key to continue ...")

if __name__ == "__main__":
    # part III
    word2vec_model = GoogleWo2VecModel(300).get_model()

    words = [
        "king", "queen", "man", "woman", "doctor", "nurse", "engineer", "scientist",
        "apple", "banana", "orange", "grape", "river", "mountain", "ocean", "forest",
        "dog", "cat", "bird", "fish", "car", "truck", "bicycle", "train", "airplane",
        "computer", "keyboard", "mouse", "screen", "love", "happiness", "sadness", 
        "anger", "joy", "pencil", "book", "notebook", "desk", "chair", "guitar", 
        "piano", "violin", "music", "song", "melody", "flower", "tree", "grass", 
        "bush", "cloud", "sun", "moon", "star", "planet", "galaxy", "comet", "saturn",
        "earth", "mercury", "venus", "mars", "jupiter", "neptune", "pluto", "athlete",
        "basketball", "soccer", "tennis", "football", "swimming", "running", "cycling",
        "painting", "drawing", "sculpture", "dancing", "singing"
    ]

    words = [word for word in words if word in word2vec_model]
    word_vectors = np.array([word2vec_model[word] for word in words])

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(word_vectors)
    labels = kmeans.labels_
    
    tsne = TSNE(n_components=2, random_state=42)
    word_vectors_2d = tsne.fit_transform(word_vectors)

    plt.figure(figsize=(12, 8))
    colors = ["red", "blue", "yellow"]

    for i, word in enumerate(words):
        plt.scatter(word_vectors_2d[i, 0], word_vectors_2d[i, 1], color=colors[labels[i]])
        plt.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]), fontsize=9)


    plt.title("3-Cluster Analysis of 75 Words")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show(block=False)

    clustered_words = {i: [] for i in range(3)}
    for i, word in enumerate(words):
        clustered_words[labels[i]].append(word)

    print("Clustered Words:")
    for cluster, words in clustered_words.items():
        print(f"Cluster {cluster + 1}: {words}")

    input("Press any key to continue ...")