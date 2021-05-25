import json
import logging
import os

import numpy as np
from tqdm import tqdm

from src.Document import Document
from src.SimilarityScorer import SimilarityScorer
from src.TfIdfVectorizer import TfIdfVectorizer


class QuestionsSearchEngine:
    def __init__(self, questions: list, stop_words_path="", embedding_size=100) -> None:
        """Initialize search engine by vectorizing question corpus.
        Input questions should be used to fit the TF-IDF vectorizer.
        Vectorized question corpus should be used to find the top n most
        similar questions w.r.t. input query.
        Args:
        questions: The sequence of raw questions from corpus.
        """
        self._vectorizer = TfIdfVectorizer(stop_words_path=stop_words_path, embedding_size=embedding_size)
        question_list = []
        for document in questions:
            document: Document
            question_list.append(document.text)

        self._vectorizer.fit(documents=question_list)
        logging.log(logging.INFO, "Finished model fitting")
        vector_matrix = self._vectorizer.transform(documents=question_list, progressbar=True)
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
        ):
        # List[Tuple[float, str]]
        """Return top n most similar questions from corpus.
        Input question should be cleaned and vectorized with fitted
        TfIdfVectorizer to get query question vectors. After that, use
        cosine_similarity function to get the top n most similar
        questions from the corpus.
        Args:
        query: The raw query question input from the user.
        n: The number of similar questions returned from corpus.
        Returns:
        The list of top n most similar questions from corpus along
        with similarity scores. Note that returned questions are
        verbatim.
        """
        sim_scorer = SimilarityScorer()
        query_vector = self._vectorizer.transform(documents=[query])
        similarity_scores = sim_scorer.cosine_similarity(query_vectors=query_vector, corpus_vectors=self._stored_data_vectors)
        similarity_scores = similarity_scores.T
        best_scores = (similarity_scores.argsort(axis=1))
        best_n_scores = best_scores[:, -n:]
        query_result = []
        for index in best_n_scores[0]:
            document: Document = self._stored_data[index]
            query_result.append((similarity_scores[0, index].round(decimals=4), document.text))
        return query_result

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
        return documents


if __name__ == "__main__":
    test_questions = [
        "how to make sure a file's integrity in C#",
        "c# index was out of the bounds of the array",
        "MySQL how to query five tables in one SELECT",
        "Add description attribute to enum and read this description in TypeScript"
    ]
    documents = QuestionsSearchEngine.load_questions(path="data/questions.jsonl")
    # documents = documents[:1000]
    qse = QuestionsSearchEngine(questions=documents, stop_words_path="data/stop_words_english.json")
    for t_question in test_questions:
        print(t_question)
        print('-'*20)
        r_query = qse.most_similar(query=t_question, n=5)
        for rq in r_query:
            print(rq)
        print('\n\n\n')
