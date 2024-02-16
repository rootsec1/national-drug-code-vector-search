from gensim.models import Word2Vec

class TextEmbedder:
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def train_model(self, data):
        all_text = []
        for match in data:
            for field, value in match.items():
                if isinstance(value, str):
                    all_text.append(value)
        self.model = Word2Vec(sentences=all_text, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers)

    def create_embeddings(self, data):
        if self.model is None:
            raise ValueError("You must train the model before creating embeddings.")
        for match in data:
            text_fields = [value for field, value in match.items() if isinstance(value, str)]
            text_embeddings = [self.model.wv[word] for word in text_fields]
            combined_embedding = sum(text_embeddings) / len(text_embeddings)
            match['text_embedding'] = combined_embedding.tolist()

#embedder = TextEmbedder()
###embedder.train_model(data)
#mbedder.create_embeddings(data)
