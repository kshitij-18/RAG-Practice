from chunkers.MarkdownTextSplitter import MarkdownTextSplitter
from VectorDB.PineConeVectorDb import PineconeIndexDb
from embedders.BM25Embedder import Bm25Embedder
from embedders.DenseEmbedder import DenseEmbedder
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.retrievers import PineconeHybridSearchRetriever

class StoragePipeline:
    def __init__(self, collection_name: str, docs_path: str) -> None:
        self.collection_name = collection_name
        self.docs_path = docs_path
        self.text_splitter = MarkdownTextSplitter(self.docs_path)
        self.index = PineconeIndexDb.create_instance(collection_name=collection_name)
    
    def store(self):
        chunks = self.text_splitter.get_chunks()
        dense_ebedder = DenseEmbedder(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        bm25_embedder = Bm25Embedder()
        bm25_embedder.fit_documents([doc.page_content for doc in chunks])
        bm25_embedder.dump_values("bm25_vectors.json")
        
        hybrid_retriever = PineconeHybridSearchRetriever(
        index=self.index,
        embeddings=dense_ebedder.model,
        sparse_encoder=bm25_embedder.model,
        )

        hybrid_retriever.add_texts(texts=[doc.page_content for doc in chunks], metadatas=[doc.metadata for doc in chunks])



        
