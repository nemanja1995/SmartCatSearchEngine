"""

"""

import argparse
import os

from src.QuestionSearchEngine import QuestionsSearchEngine

parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--config', default='config.json', help="Path to config file")
parser.add_argument('-v', '--vector_size', default=100, help="Size of embedding vectors")
parser.add_argument('-d', '--corpus_path', default='data/questions.jsonl',
                    help="Path to corpus", required=False)
parser.add_argument('-s', '--stop_words_path', default='data/stop_words_english.json',
                    help="Path to file with stopwords")
parser.add_argument('-f', '--force_process', default=False,
                    help="Force engine to process documents")
parser.add_argument('-dp', '--qse_data_path', default='cached/qse_data.pkl',
                    help="Force engine to process documents")
args = parser.parse_args()


def main():
    print(args)
    test_questions = [
        "how to make sure a file's integrity in C#",
        "c# index was out of the bounds of the array",
        "MySQL how to query five tables in one SELECT",
        "Add description attribute to enum and read this description in TypeScript"
    ]

    qse = None
    force_process = args.force_process
    qse_data_path = args.qse_data_path
    if os.path.exists(qse_data_path) and not force_process:
        qse = QuestionsSearchEngine(skip_process=True)
        qse.load_stored_data(qse_data_path)

    else:
        documents = QuestionsSearchEngine.load_questions(path="data/questions.jsonl")
        qse = QuestionsSearchEngine(questions=documents, stop_words_path="data/stop_words_english.json")

    for t_question in test_questions:
        print(t_question)
        print('-'*20)
        r_query = qse.most_similar(query=t_question, n=5)
        for rq in r_query:
            print(rq)
        print('\n\n\n')

    qse.save_stored_data(path=qse_data_path)


if __name__ == "__main__":
    main()
