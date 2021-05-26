"""
Implemented Question Search Engine witch search similar question based on vector similarity.
"""
import json
import logging
import os
import pickle


from src.Document import Document
from src.SimilarityScorer import SimilarityScorer
from src.TfIdfVectorizer import TfIdfVectorizer


class QuestionsSearchEngine:
    STORED_INFO_FILE = 'info_data'
    STORED_VECTORS_FILE = 'vectors'

    def __init__(self, questions=None, stop_words_path="", embedding_size=100, skip_process=False) -> None:
        """
        Initialize search engine by vectorizing question corpus.
        :param questions:
        :param stop_words_path:
        :param embedding_size:
        :param skip_process:
        """
        if questions is None:
            questions = []
        if skip_process:
            return

        self._vectorizer = TfIdfVectorizer(stop_words_path=stop_words_path, embedding_size=embedding_size)
        question_list = []
        for document in questions:
            document: Document
            question_list.append(document.text)

        self._vectorizer.fit(questions=question_list)
        logging.log(logging.INFO, "Finished model fitting")
        vector_matrix = self._vectorizer.transform(questions=question_list)
        logging.log(logging.INFO, "Finished processing corpus into vectors")
        for doc, vector in zip(questions, vector_matrix[:]):
            doc: Document
            doc.vector = vector

        self._stored_data = questions
        self._stored_data_vectors = vector_matrix

    def most_similar(
        self,
        query: str,
        n: int = 5
        ) -> list:
        """
        Return top n most similar questions from corpus.
        Input question is cleaned and vectorized with fitted
        TfIdfVectorizer. After that, use
        cosine_similarity function to get the top n most similar
        questions from the corpus.
        :param query: The raw query question input from the user.
        :param n: The number of similar questions returned from corpus.
        :return: The list of top n most similar questions from corpus along
        with similarity scores. Note that returned questions are
        verbatim.
        """

        sim_scorer = SimilarityScorer()

        # Transform query question into vector
        self._vectorizer.progress_bar = False
        query_vector = self._vectorizer.transform(questions=[query])
        self._vectorizer.progress_bar = True

        # Search similar question with cosine similarity
        similarity_scores = sim_scorer.cosine_similarity(query_vectors=query_vector, corpus_vectors=self._stored_data_vectors)
        similarity_scores = similarity_scores.T

        # Find N most similar questions from corpus
        best_scores = (similarity_scores.argsort(axis=1))
        best_n_scores = best_scores[:, -n:]

        query_result = []
        for index in best_n_scores[0]:
            document: Document = self._stored_data[index]
            query_result.append((similarity_scores[0, index].round(decimals=4), document.text))
        return query_result

    def save_stored_data(self, path):
        """
        Save processed data on disk
        :param path: Path to file where to save cached data.
        """
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(path=dirname):
            os.makedirs(dirname)

        if not os.path.basename(path):
            path = os.path.join(path, 'qse_data.pkl')

        with open(path, 'wb') as output:
            pickle.dump(self._stored_data, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self._stored_data_vectors, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self._vectorizer, output, pickle.HIGHEST_PROTOCOL)

    def load_stored_data(self, path):
        """
        Loads cached data for Question Searched Engine
        :param path: Path to file where is cached data stored.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError("Given path to Question Search Engine cache does not exist.")

        with open(path, 'rb') as input:
            self._stored_data = pickle.load(input)
            self._stored_data_vectors = pickle.load(input)
            self._vectorizer = pickle.load(input)

    @staticmethod
    def load_questions(path):
        if not os.path.exists(path):
            return
        fr = open(path, 'r')
        documents = []
        for row in fr.readlines():
            row = row.replace('\n', '')
            sentence_json = json.loads(row)
            question = sentence_json.get('question', '')
            doc_id = sentence_json.get('id', '')
            tags = sentence_json.get('tags', '')
            doc = Document(text=question, doc_id=doc_id, tags=tags)
            documents.append(doc)
        fr.close()
        return documents


if __name__ == "__main__":
    # test_questions = [
    #     "how to make sure a file's integrity in C#",
    #     "c# index was out of the bounds of the array",
    #     "MySQL how to query five tables in one SELECT",
    #     "Add description attribute to enum and read this description in TypeScript"
    # ]
    # documents = QuestionsSearchEngine.load_questions(path="data/questions.jsonl")
    # # documents = documents[:1000]
    # qse = QuestionsSearchEngine(questions=documents, stop_words_path="data/stop_words_english.json")
    # for t_question in test_questions:
    #     print(t_question)
    #     print('-'*20)
    #     r_query = qse.most_similar(query=t_question, n=5)
    #     for rq in r_query:
    #         print(rq)
    #     print('\n\n\n')
    pass
