from embedders.BaseEmbedder import BaseEmbedder
from abc import ABC, abstractmethod
from typing import Any
from pinecone_text.sparse.base_sparse_encoder import BaseSparseEncoder
class SparseEmbedder(ABC):
    def __init__(self, model: BaseSparseEncoder):
        super().__init__()
        self._model = model

    @property
    def model(self):
        return self._model

    @abstractmethod
    def fit_documents(self, documents: list[str]):
        pass

    @abstractmethod
    def dump_values(self, file_to_dump: str):
        pass
