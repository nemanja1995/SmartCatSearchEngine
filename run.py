"""
 Run script for
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
                    help="Force engine to process corpus again")
parser.add_argument('-dp', '--qse_data_path', default='cached/qse_data.pkl',
                    help="Path to cached data for question search engine")
args = parser.parse_args()


def main():
    print(args)

    force_process = args.force_process
    qse_data_path = args.qse_data_path
    stop_words_path = args.stop_words_path
    vector_size = args.vector_size
    corpus_path = args.corpus_path

    loaded_from_cache = False
    if os.path.exists(qse_data_path) and not force_process:
        qse = QuestionsSearchEngine(skip_process=True)
        qse.load_stored_data(qse_data_path)
        loaded_from_cache = True

    else:
        documents = QuestionsSearchEngine.load_questions(path=corpus_path)
        qse = QuestionsSearchEngine(questions=documents,
                                    stop_words_path=stop_words_path,
                                    embedding_size=vector_size)

    finished = False
    while not finished:
        print("> enter query: ")
        question = input()
        if type(question) != str:
            print("> Wrong type entered")
            continue

        if question.lower() in ['quit', 'exit', 'end', 'done']:
            finished = True
            break
        print('-'*20)
        r_query = qse.most_similar(query=question, n=5)
        for rq in r_query:
            print(rq)
        print('\n\n')

    if not loaded_from_cache:
        qse.save_stored_data(path=qse_data_path)


if __name__ == "__main__":
    main()
