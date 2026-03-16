from chunkers.BaseChunker import BaseChunker
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from datetime import date


class MarkdownTextSplitter(BaseChunker):
    def __init__(self, docs_path: str):
        super().__init__(docs_path)

    def get_chunks(self):
        chunks = []
        for doc in os.listdir(self.docs_path):
            try:
                # Try UTF-8 first, fall back to latin-1 if it fails
                try:
                    doc_content = TextLoader(os.path.join(self.docs_path, doc), encoding="utf-8").load()
                except (UnicodeDecodeError, LookupError):
                    print(f"⚠️  UTF-8 failed for {doc}, trying latin-1...")
                    doc_content = TextLoader(os.path.join(self.docs_path, doc), encoding="latin-1").load()

                # chunk the data using MarkdownHeaderTextSplitter
                headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
                ("#####", "Header 5"),
                ("######", "Header 6"),]
                text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                doc_chunks = text_splitter.split_text(doc_content[0].page_content)
                for doc_chunk in doc_chunks:
                    meta_data = {
                        "source": doc,
                        "chunk_content": doc_chunk.page_content,
                        "timestamp": date.today().ctime(),
                        **doc_chunk.metadata
                    }
                    chunk = Document(page_content=doc_chunk.page_content, metadata=meta_data)
                    chunks.append(chunk)
                print(f"✓ Loaded {doc}: {len(doc_chunks)} chunks")
            except Exception as e:
                print(f"❌ Error loading {doc}: {str(e)}")
                continue

        print(f"\n✓ Total chunks loaded: {len(chunks)}")
        return chunks