from embedders.BaseEmbedder import BaseEmbedder
from langchain_core.embeddings.embeddings import Embeddings
class DenseEmbedder(BaseEmbedder):
    def __init__(self, model: Embeddings):
        super().__init__()
        self._model = model
    
    def embed_documents(self, documents: list[str]):
        return self.model.embed_documents(documents)
    
    @property
    def model(self):
        return self._model