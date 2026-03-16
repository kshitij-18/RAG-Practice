from chunkers.MarkdownTextSplitter import MarkdownTextSplitter
from VectorDB.PineConeVectorDb import PineconeHybridRetrieverVectorDB

class StoragePipeline:
    def __init__(self, collection_name: str, docs_path: str) -> None:
        self.collection_name = collection_name
        self.docs_path = docs_path
        self.vector_db = PineconeHybridRetrieverVectorDB(collection_name)
        self.text_splitter = MarkdownTextSplitter(self.docs_path)
    
    def store(self):
        chunks = self.text_splitter.get_chunks()
        self.vector_db.add_documents(chunks)
