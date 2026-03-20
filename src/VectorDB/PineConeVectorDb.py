from pinecone import Pinecone, ServerlessSpec
import os
from embedders.DenseEmbedder import DenseEmbedder
from embedders.BM25Embedder import Bm25Embedder
from pinecone_text.sparse.bm25_encoder import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


class PineconeIndexDb:
    pinecone_index = None
    def __init__(self):
        raise Exception('Please do not instantiate this class use the create_instance class')
        

    @staticmethod
    def create_instance(collection_name: str):
        if not PineconeIndexDb.pinecone_index:
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
            
            index = pc.Index(collection_name)
            PineconeIndexDb.pinecone_index = index
        return PineconeIndexDb.pinecone_index
