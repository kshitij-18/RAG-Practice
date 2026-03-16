from abc import ABC, abstractmethod

class BasevectorDB(ABC):
    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    @abstractmethod
    def add_documents(self, documents: list):
        pass