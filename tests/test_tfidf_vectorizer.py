import numpy as np
import unittest

from src.TfIdfVectorizer import TfIdfVectorizer


class TfIdfVectorizerTestCase(unittest.TestCase):
    CORPUS_PATH = "../data/questions.jsonl"

    def test_tf_idf(self):
        documents = TfIdfVectorizer.load_questions(path=TfIdfVectorizerTestCase.CORPUS_PATH)
        documents = documents[:1000]
        vectorizer = TfIdfVectorizer(embedding_size=100, progress_bar=False)
        vectorizer.fit(documents)
        question = 'How to write Function in Python with list?'

        word_list = question.split()
        words = list(set(word_list))
        for word in words:
            tf_idf_score, _ = vectorizer.tf_idf_info(word=word, document=question, word_list=word_list)
            valid_type = type(tf_idf_score) is np.float64
            self.assertTrue(valid_type, "TF IDF score should be float type")

    def test_vectorizer(self):
        test_documents = [
            "How to write Function in Python with list?",
            "What is array in JavaScript?"
        ]

        documents = TfIdfVectorizer.load_questions(path=TfIdfVectorizerTestCase.CORPUS_PATH)
        documents = documents[:1000]
        embedding_size = 100
        vectorizer = TfIdfVectorizer(embedding_size=embedding_size, progress_bar=False)

        vectorizer.fit(questions=documents)

        result = vectorizer.transform(questions=test_documents)
        self.assertTrue(type(result) is np.ndarray)
        self.assertTrue(result.shape == (len(test_documents), embedding_size))
        for doc, vec in zip(test_documents, result):
            arr_sum = vec.sum()
            self.assertTrue(arr_sum > 0, "All values in vector are zeros")


if __name__ == '__main__':
    unittest.main()
