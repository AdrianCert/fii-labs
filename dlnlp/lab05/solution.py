import requests
import nltk
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation


# Download NLTK resources if not already downloaded
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords


class WikipediaClient:
    """A simple Wikipedia client to search for pages and fetch content using the Wikipedia API."""

    WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
    CACHE_DIR = Path.home() / ".cache/wikipedia_client"

    def search_topic(self, topic, limit=10):
        """Searches for Wikipedia pages related to a topic and returns a list of page titles."""
        params = {
            "action": "opensearch",
            "format": "json",
            "search": topic,
            "limit": limit,
        }
        response = requests.get(self.WIKI_API_URL, params=params)
        data = response.json()

        return data[1]

    def fetch_page(self, page_title):
        """Fetches the text of a Wikipedia page by title using the Wikipedia API."""
        _cache_file = self.CACHE_DIR / f"pages/{page_title}.txt"
        if _cache_file.exists():
            return _cache_file.read_text(encoding="utf-8")
        params = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "extracts",
            "explaintext": True,
        }
        response = requests.get(self.WIKI_API_URL, params=params)
        data = response.json()
        page: dict = next(iter(data["query"]["pages"].values()))
        page_extract = page.get("extract")
        if not page_extract:
            raise ValueError("Page not found")
        _cache_file.parent.mkdir(parents=True, exist_ok=True)
        _cache_file.write_text(page_extract, encoding="utf-8")
        return page_extract

    def fetch_topic_pages(self, topic, limit=10):
        """Fetches the text of Wikipedia pages related to a topic."""
        page_titles = self.search_topic(topic, limit)
        pages = {}
        for title in page_titles:
            try:
                text = self.fetch_page(title)
                pages[title] = text
            except ValueError as _e:
                print(f"Error fetching page: {title}")
        return pages


class TextProcessing:
    def __init__(self):
        self.ensure_resources()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def ensure_resources(self):
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")

    def process_text_iter(self, text, use_stemming=False, use_lemmatization=False):
        words = nltk.word_tokenize(text)

        for word in words:
            word = word.lower()
            if not word.isalpha() or word in self.stop_words:
                continue
            if use_stemming:
                word = self.stemmer.stem(word)
            if use_lemmatization:
                word = self.lemmatizer.lemmatize(word)
            yield word

    def process_text(self, text, use_stemming=True, use_lemmatization=False):
        return " ".join(self.process_text_iter(text, use_stemming, use_lemmatization))


class IfIdProcessing:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.bow_vectorizer = CountVectorizer()

    def create_term_document_matrix(self, docs):
        return self.vectorizer.fit_transform(docs)

    def create_bag_of_words_matrix(self, docs):
        return self.bow_vectorizer.fit_transform(docs)


class Visualization:
    """Class to visualize LSA and NMF results."""

    @staticmethod
    def visualize(reduced_matrix, labels, method="LSA"):
        """Visualizes the reduced data in 2D or 3D."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=reduced_matrix[:, 0],
            y=reduced_matrix[:, 1],
            hue=labels,
            palette="viridis",
        )
        plt.title(f"2D Visualization of {method}-Reduced Data")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(title="Topics")
        plt.show()

    @staticmethod
    def visualize_3d(reduced_matrix, labels, method="LSA"):
        """Visualizes the reduced data in 3D."""
        import plotly.express as px

        fig = px.scatter_3d(
            x=reduced_matrix[:, 0],
            y=reduced_matrix[:, 1],
            z=reduced_matrix[:, 2]
            if reduced_matrix.shape[1] > 2
            else np.zeros(reduced_matrix.shape[0]),
            color=labels,
            title=f"3D Visualization of {method}-Reduced Data",
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3",
            )
        )
        fig.show()


class BaseProcessing:
    def apply(self, matrix): ...

    def visualize(self, reduced_matrix, labels):
        Visualization.visualize(reduced_matrix, labels, method=self.__class__.__name__)

    def visualize_3d(self, reduced_matrix, labels):
        Visualization.visualize_3d(
            reduced_matrix, labels, method=self.__class__.__name__
        )


class LSAProcessing(BaseProcessing):
    """Class to perform Latent Semantic Analysis on BoW and TF-IDF encodings using SVD."""

    def __init__(self, n_components=3):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components)

    def apply(self, matrix):
        """Applies SVD to reduce dimensions of the term-document matrix."""
        reduced_matrix = self.svd.fit_transform(matrix)
        return reduced_matrix


class NMFProcessing(BaseProcessing):
    """Class to perform Non-negative Matrix Factorization (NMF) on BoW and TF-IDF encodings."""

    def __init__(self, n_components=3):
        self.n_components = n_components
        self.nmf = NMF(n_components=n_components, random_state=43)

    def apply(self, matrix):
        """Applies NMF to reduce dimensions of the term-document matrix."""
        reduced_matrix = self.nmf.fit_transform(matrix)
        return reduced_matrix


class LDAProcessing(BaseProcessing):
    """Class to perform Latent Dirichlet Allocation (LDA) on text data."""

    def __init__(self, num_topics=3):
        self.num_topics = num_topics
        self.lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)

    def apply(self, texts):
        """Applies LDA to find topics in the text data."""
        self.lda.fit(texts)
        return self.lda.transform(texts)


class LDA2Processing(BaseProcessing):
    """Class to perform Latent Dirichlet Allocation (LDA) on text data."""

    def __init__(self, num_topics=3):
        import gensim
        import gensim.corpora as corpora

        self.corpora = corpora
        self.gensim = gensim
        self.num_topics = num_topics

    def apply(self, texts):
        self.id2word = self.corpora.Dictionary(texts)
        self.corpus = list(map(self.id2word.doc2bow, texts))
        self.lda_model = self.gensim.models.LdaMulticore(
            corpus=self.corpus,
            id2word=self.id2word,
            num_topics=self.num_topics,
        )
        return self.lda_model

    def visualize(self, reduced_matrix, labels):
        import pyLDAvis.gensim

        vis = pyLDAvis.gensim.prepare(self.lda_model, self.corpus, self.id2word)
        pyLDAvis.save_html(vis, "lda.html")


if __name__ == "__main__":
    topics = ["Tennis", "Art", "Travel"]
    wikipedia_client = WikipediaClient()
    text_processing = TextProcessing()
    if_id_processing = IfIdProcessing()
    lsa_processing = LSAProcessing()

    pages = list(
        itertools.chain(
            *[
                wikipedia_client.fetch_topic_pages(topic, limit=5).values()
                for topic in topics
            ]
        )
    )
    processed_pages = [text_processing.process_text(page) for page in pages]

    matrix_idf = if_id_processing.create_term_document_matrix(processed_pages)
    matrix_bow = if_id_processing.create_bag_of_words_matrix(processed_pages)
    labels = list(itertools.chain(*[[topic] * 5 for topic in topics]))

    # Visualize using LSA
    # lda2_processing = LDA2Processing()
    # lda2_processing.apply([list(text_processing.process_text_iter(page)) for page in pages])
    # lda2_processing.visualize(None, labels)
    # input("Press Enter to continue...")

    # Visualize the reduced matrices
    for method in [LSAProcessing, NMFProcessing, LDAProcessing]:
        processor: BaseProcessing = method()
        reduced_idf = processor.apply(matrix_idf)
        processor.visualize_3d(reduced_idf, labels)
        reduced_bow = processor.apply(matrix_bow)
        processor.visualize_3d(reduced_bow, labels)
