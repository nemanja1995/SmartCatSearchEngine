

class Document:
    def __init__(self, doc_id, text, tags, vector=None):
        self.vector = vector
        self.doc_id = doc_id
        self.text = text
        self.tags = tags
