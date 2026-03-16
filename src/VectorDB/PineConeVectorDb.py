from VectorDB.BaseVectorDB import BasevectorDB
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
import os
from embedders.DenseEmbedder import DenseEmbedder
from embedders.BM25Embedder import Bm25Embedder
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse.bm25_encoder import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


class PineconeHybridRetrieverVectorDB(BasevectorDB):
    def __init__(self, collection_name: str):
        super().__init__(collection_name)
        pc = Pinecone(api_key=os.getenv('PINECONE_KEY'))
        if pc.has_index(collection_name):
            print("Deleting old index with incorrect dimension...")
            pc.delete_index(collection_name)
    
        if not pc.has_index(collection_name):
            print(f"Creating new index '{collection_name}' with dimension 384...")
            pc.create_index(
                name=collection_name,
                dimension=384,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        
        self.index = pc.Index(collection_name)
        stats = self.index.describe_index_stats()
        if stats['total_vector_count'] > 0:
            print(f"Index '{collection_name}' already has {stats['total_vector_count']} vectors.")

    def add_documents(self, documents: list[Document]):
        dense_ebedder = DenseEmbedder(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        bm25_embedder = Bm25Embedder(BM25Encoder())
        bm25_embedder.fit_documents([doc.page_content for doc in documents])
        bm25_embedder.dump_values("bm25_vectors.json")

        hybrid_retriever = PineconeHybridSearchRetriever(
            index=self.index,
            embeddings=dense_ebedder.model,
            sparse_encoder=bm25_embedder.model,
        )

        hybrid_retriever.add_texts(texts=[doc.page_content for doc in documents], metadatas=[doc.metadata for doc in documents])




