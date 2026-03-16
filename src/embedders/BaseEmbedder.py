from abc import ABC, abstractmethod

class BaseEmbedder(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def embed_documents(self, documents: list[str]):
        pass