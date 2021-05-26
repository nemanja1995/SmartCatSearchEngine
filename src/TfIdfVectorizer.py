"""
Implementation of Sentence embedding based on bags-of-words and TfIdf algorithm.

"""
import json
import os
import re
import numpy as np
from tqdm import tqdm


class TfIdfVectorizer:
    def __init__(self, stop_words_path="", embedding_size=1000, progress_bar=True):
        self.progress_bar = progress_bar

        # Full processed dictionary of all words in corpus
        self._word_count_dict = {}
        self._total_corpus_size = -1

        # Subset of full dictionary, it should contain only embedding size number of words
        # key - word
        # value - (word_index, appearance_count)
        # TODO: Create getter for word_index and appearance_count
        self._bag_word_vocabulary = {}
        self._embedding_size = embedding_size

        if stop_words_path:
            self.load_stop_words(path=stop_words_path)
        else:
            self._stop_words = []

    def load_stop_words(self, path):
        if not os.path.exists(path):
            return

        with open(path, 'r') as fr:
            stop_words = json.load(fr)
        self._stop_words = stop_words

    @staticmethod
    def load_questions(path):
        fr = open(path, 'r')
        documents = []
        for row in fr.readlines():
            row = row.replace('\n', '')
            sentence_json = json.loads(row)
            documents.append(sentence_json['question'])
        fr.close()
        return documents

    def set_and_sort_word_dict(self, word_count_dict: dict):
        """
        Sorts given dictionary and set internal word count dictionary
        :param word_count_dict: Processed dictionary with every word appearance count
        """
        # If stop word list is loaded, clear stop words
        for stop_word in self._stop_words:
            if stop_word in word_count_dict:
                word_count_dict.pop(stop_word)

        # Sort dictionary by appearance count
        sorted_word_count_dict = {}
        for word in sorted(word_count_dict, key=word_count_dict.get, reverse=True):
            sorted_word_count_dict[word] = word_count_dict[word]

        self._word_count_dict = sorted_word_count_dict
        self._total_corpus_size = len(sorted_word_count_dict)
        self._bag_word_vocabulary = self.get_first_n_words(n=self._embedding_size)

    def get_first_n_words(self, n: int) -> dict:
        """
        Returns most frequent words N words as dictionary
        :param n: Number of words
        :return: dictionary - (word_index, appearance_count)
        """
        firs_n_words_dict = {}
        for word_index, (key, app_count) in enumerate(self._word_count_dict.items()):
            if word_index >= n:
                break
            # TODO: Change this structure it is not really readable
            firs_n_words_dict[key] = (word_index, app_count)
        return firs_n_words_dict

    @staticmethod
    def trim_string(tmp_string: str) -> str:
        """
        Lower case and trim non words characters
        :param tmp_string: Any string
        :return: Trimmed string
        """
        tmp_string = tmp_string.lower()
        # Clear all non word characters
        trimmed_string = re.sub(r'[\d\W]+', ' ', tmp_string)
        return trimmed_string

    def fit(self, questions):
        """Fit vectorizer with the sequence of documents (questions), after this vectorizer can be used for transforming
        sentences into vectors.
        """
        word_count_dict = {}
        for doc in tqdm(questions, desc="Fitting vectorizer model", disable= not self.progress_bar):
            cl_doc = TfIdfVectorizer.trim_string(doc)
            words = list(set(cl_doc.split()))
            for word in words:
                word_count_dict[word] = word_count_dict.get(word, 0) + 1
        self.set_and_sort_word_dict(word_count_dict=word_count_dict)

    def tf_idf_info(self, word: str, document: str, word_list=None) -> (float, int):
        """
        Calculate TF-IDF score for given word and document with already fit-ed corpus.
        :param word: Any word from document
        :param document: any string
        :param word_list: Instead of document it can use already processed list of words from document
        :return: (tf_idf_score, found_in_corpus), found_in_corpus - True if word is in bag-of-words corpus
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
        Wrapper for tf_idf_info, if simpler method signature is needed
        """
        tf_idf_score, _ = self.tf_idf_info(word=word, document=document)
        return tf_idf_score

    def transform_doc(self, question) -> np.ndarray:
        """
        Transform texts into numpy vectors with TF-IDF scores.
        :return: Vectorized questions as numpy array of (N, D) shape where
                N is number of questions in corpus, and D is vocabulary size
                Used in a bag-of-words model.
        """
        cl_sentence = TfIdfVectorizer.trim_string(question)
        word_list = cl_sentence.split()

        # Clear multiple word appearance from word list
        words = list(set(word_list))
        embedding = np.zeros(self._embedding_size, dtype=float)

        for word in words:
            tf_idf_score, found_in_corpus = self.tf_idf_info(word=word, document=question, word_list=word_list)
            if found_in_corpus:
                word_index = self._bag_word_vocabulary.get(word)[0]
                embedding[word_index] = tf_idf_score
        return embedding

    def transform(self, questions) -> np.ndarray:
        """
        Transform texts into numpy vectors with TF-IDF scores.
        :param questions: The sequence of raw corpus questions.
        :return: Vectorized questions as numpy array of (N, D) shape where
                N is number of questions in corpus, and D is vocabulary size
                Used in a bag-of-words model.
        """
        vectors = np.zeros(shape=(0, self._embedding_size), dtype=float)
        for num, document in tqdm(enumerate(questions), desc="Processing documents into vectors",
                                  total=len(questions), disable=not self.progress_bar):
            vector = self.transform_doc(question=document)
            vectors = np.vstack([vectors, vector])
        return vectors

    def save_bag_word_dict(self, path):
        json.dump(self._bag_word_vocabulary, open(path, 'w'), indent=4)

