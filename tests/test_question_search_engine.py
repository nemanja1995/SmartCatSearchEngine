import numpy as np
import unittest

from src.QuestionSearchEngine import QuestionsSearchEngine


class QuestionSearchEngineTestCase(unittest.TestCase):
    CORPUS_PATH = "../data/questions.jsonl"

    def test_search(self):
        test_questions = [
            "how to make sure a file's integrity in C#",
            "c# index was out of the bounds of the array",
            "MySQL how to query five tables in one SELECT",
            "Add description attribute to enum and read this description in TypeScript"
        ]
        n = 5
        documents = QuestionsSearchEngine.load_questions(path=QuestionSearchEngineTestCase.CORPUS_PATH)
        documents = documents[:1000]
        qse = QuestionsSearchEngine(questions=documents)

        for t_question in test_questions:
            r_query = qse.most_similar(query=t_question, n=n)
            self.assertTrue(len(r_query) == 5)
            for rq in r_query:
                similarity = rq[0]
                self.assertTrue(type(similarity) is np.float64)
                self.assertTrue(-1 <= similarity <= 1)


if __name__ == '__main__':
    unittest.main()
