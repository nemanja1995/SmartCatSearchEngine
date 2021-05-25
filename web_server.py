# Python 3 server example
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import time

from src.QuestionSearchEngine import QuestionsSearchEngine

hostName = "localhost"
serverPort = 8081

qse_data_path = 'cached/qse_data.pkl'
if os.path.exists(qse_data_path):
    qse = QuestionsSearchEngine(skip_process=True)
    qse.load_stored_data(qse_data_path)
else:
    documents = QuestionsSearchEngine.load_questions(path="data/questions.jsonl")
    qse = QuestionsSearchEngine(questions=documents, stop_words_path="data/stop_words_english.json")


class MyServer(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html><head><title>https://Smart Cat Search</title></head>", "utf-8"))
        self.wfile.write(bytes("<p>Request: %s</p>" % self.path, "utf-8"))
        self.wfile.write(bytes("<body>", "utf-8"))
        self.wfile.write(bytes("<p>This is an example web server for testing smart cat search.</p>", "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                str(self.path), str(self.headers), post_data.decode('utf-8'))

        post_body = post_data.decode()
        data_dict = json.loads(post_body)
        self._set_response()

        ## Do some processing
        questions = data_dict.get('questions', None)
        results = []
        if questions and type(questions) is list:
            for t_question in questions:
                if type(t_question) is str:
                    r_query = qse.most_similar(query=t_question, n=5)
                    results.append({"question": t_question, "similar_questions": r_query})
        ## Reprocess data

        data = json.dumps(results)
        self.wfile.write(data.encode('utf-8'))


if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
