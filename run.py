"""

"""

import argparse

from src.QuestionSearchEngine import QuestionsSearchEngine

parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--config', default='config.json', help="Path to config file")
parser.add_argument('-v', '--vector_size', default=100, help="Size of embedding vectors")
parser.add_argument('-d', '--corpus_path', default='data/questions.jsonl',
                    help="Path to corpus", required=False)
parser.add_argument('-s', '--stop_words_path', default='data/stop_words_english.json',
                    help="Path to file with stopwords")
args = parser.parse_args()


def main():
    print(args)
    test_questions = [
        "how to make sure a file's integrity in C#",
        "c# index was out of the bounds of the array",
        "MySQL how to query five tables in one SELECT",
        "Add description attribute to enum and read this description in TypeScript"
    ]
    documents = QuestionsSearchEngine.load_questions(path="data/questions.jsonl")
    documents = documents[:1000]
    qse = QuestionsSearchEngine(questions=documents, stop_words_path="data/stop_words_english.json")
    for t_question in test_questions:
        print(t_question)
        print('-'*20)
        r_query = qse.most_similar(query=t_question, n=5)
        for rq in r_query:
            print(rq)
        print('\n\n\n')


if __name__ == "__main__":
    main()
