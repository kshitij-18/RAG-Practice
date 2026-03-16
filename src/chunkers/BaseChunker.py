from abc import ABC, abstractmethod

class BaseChunker(ABC):
    def __init__(self, docs_path: str):
        self.docs_path = docs_path

    @abstractmethod
    def get_chunks(self, *args, **kwargs):
        pass