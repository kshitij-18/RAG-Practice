from embedders.SparseEmbedder import SparseEmbedder
from pinecone_text.sparse.bm25_encoder import BM25Encoder
class Bm25Embedder(SparseEmbedder):
    def __init__(self):
        self._model = BM25Encoder()
        super().__init__(self._model)
    
    @property
    def model(self):
        return self._model.load(self.file_to_dump)

    def fit_documents(self, documents: list[str]):
        return self._model.fit(documents)
    
    def dump_values(self, file_to_dump: str):
        self.file_to_dump = file_to_dump
        return self._model.dump(file_to_dump)