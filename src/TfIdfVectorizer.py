"""
Implementation of Sentence embedding based on bags-of-words and TfIdf algorithm.

"""
import json
import os
import re
import numpy as np
from tqdm import tqdm


class TfIdfVectorizer:
    def __init__(self, stop_words_path="", embedding_size=1000):
        self._word_count_dict = {}
        self._total_corpus_size = -1

        self._bag_word_vocabulary = {}
        self._embedding_size = embedding_size
        if stop_words_path:
            self.load_stop_words(path=stop_words_path)
        else:
            self._stop_words = []

        self.vector_storage = []

    def load_stop_words(self, path):
        if not os.path.exists(path):
            return
        stop_words = json.load(open(path, 'r'))
        self._stop_words = stop_words

    def load(self, path):
        pass

    def save(self, path):
        pass

    @staticmethod
    def load_questions(path):
        fr = open(path, 'r')
        documents = []
        for row in fr.readlines():
            row = row.replace('\n', '')
            sentence_json = json.loads(row)
            documents.append(sentence_json['question'])
        return documents

    def set_and_sort_word_dict(self, word_count_dict: dict):
        """
        Sorts given dictionary and set internal word count dictionary
        :param embedding_size:
        :param word_count_dict:
        :return:
        """
        new_dict = {}
        for stop_word in self._stop_words:
            if stop_word in word_count_dict:
                word_count_dict.pop(stop_word)

        for word in sorted(word_count_dict, key=word_count_dict.get, reverse=True):
            new_dict[word] = word_count_dict[word]
        self._word_count_dict = new_dict
        self._total_corpus_size = len(new_dict)
        self._bag_word_vocabulary = self.get_first_n_words(n=self._embedding_size)

    def get_first_n_words(self, n: int) -> dict:
        """
        Returns sorted most frequent words
        :param n:
        :return:
        """
        firs_n_words_dict = {}
        for word_index, (key, value) in enumerate(self._word_count_dict.items()):
            if word_index >= n:
                break
            # TODO: Change this structure it is not really readable
            firs_n_words_dict[key] = (word_index, value)
        return firs_n_words_dict

    @staticmethod
    def trim_string(tmp_string: str) -> str:
        tmp_string = tmp_string.lower()
        # Clear all non word characters
        trimmed_string = re.sub(r'[\d\W]+', ' ', tmp_string)
        return trimmed_string

    def fit(self, documents):
        """Fit vectorizer with the sequence of documents (questions), after this vectorizer can be used for transforming
        sentences into vectors.
        """
        word_count_dict = {}
        for doc in tqdm(documents, desc="Fitting vectorizer model"):
            cl_doc = TfIdfVectorizer.trim_string(doc)
            words = list(set(cl_doc.split()))
            for word in words:
                word_count_dict[word] = word_count_dict.get(word, 0) + 1
        self.set_and_sort_word_dict(word_count_dict=word_count_dict)

    def tf_idf_info(self, word, document, word_list=None) -> (float, int):
        """
        Calculate TF-IDF score for given word and document with already fit-ed corpus.
        :param word: Any word from document
        :param document:
        :param word_list:
        :return:
        """
        if not self._bag_word_vocabulary:
            raise ValueError("Model should be initialized.")

        # To speedup processing
        if not word_list:
            cl_sentence = TfIdfVectorizer.trim_string(document)
            word_list = cl_sentence.split()

            if not word_list:
                return [], 0

        num_appearance = word_list.count(word)
        tf = num_appearance / len(word_list)

        num_appearance_in_corpus = self._bag_word_vocabulary.get(word, (0, 0))[1]
        idf = np.log(self._total_corpus_size / (num_appearance_in_corpus + 1)) + 1

        tf_idf_score = tf * idf
        found_in_corpus = num_appearance_in_corpus > 0
        return tf_idf_score, found_in_corpus

    def tf_idf(self, word, document):
        """
        Wrapper for simpler usage
        """
        tf_idf_score, _, _ = self.tf_idf_info(word=word, document=document)
        return tf_idf_score

    def transform_doc(self, document) -> np.ndarray:
        cl_sentence = TfIdfVectorizer.trim_string(document)
        word_list = cl_sentence.split()
        words = list(set(word_list))
        embedding = np.zeros(self._embedding_size, dtype=float)
        for word in words:
            tf_idf_score, found_in_corpus = self.tf_idf_info(word=word, document=document, word_list=word_list)
            if found_in_corpus:
                word_index = self._bag_word_vocabulary.get(word)[0]
                embedding[word_index] = tf_idf_score
        return embedding

    def transform(self, documents, progressbar=False) -> np.ndarray:
        embeddings = np.zeros(shape=(0, self._embedding_size), dtype=float)
        for num, document in tqdm(enumerate(documents), desc="Processing documents into vectors", total=len(documents), disable= not progressbar):
            embedding = self.transform_doc(document=document)
            embeddings = np.vstack([embeddings, embedding])
        return embeddings

    def save_bag_word_dict(self, path):
        json.dump(self._bag_word_vocabulary, open(path, 'w'), indent=4)


if __name__ == '__main__':
    path = 'data/questions.jsonl'
    test_documents = [
        "How to write Function in Python with list?",
        "What is array in JavaScript?"
    ]

    documents = TfIdfVectorizer.load_questions(path=path)

    vectorizer = TfIdfVectorizer(stop_words_path='data/stop_words_english.json', embedding_size=100)
    vectorizer.fit(documents=documents)

    vectorizer.save_bag_word_dict(path="data/bag_of_words.json")
    result = vectorizer.transform(documents=test_documents)
    for doc, vec in zip(test_documents, result):
        print(doc)
        print(vec)
        print("-"*50)

