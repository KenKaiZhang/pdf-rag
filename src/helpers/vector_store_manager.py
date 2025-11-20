import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
from src.configs.defaults import CHROMA_HOST, CHROMA_PORT, OLLAMA_BASE_URL


class VectorStoreManager:
    def __init__(self, collection_name: str, model_name: str, ollama_base_url: str = None):
        self.host = CHROMA_HOST
        self.port = CHROMA_PORT
        self.collection_name = collection_name
        self.client = chromadb.HttpClient(
            host=self.host,
            port=self.port,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Use Ollama embeddings instead of HuggingFace
        self.embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=ollama_base_url or OLLAMA_BASE_URL
        )
        self.vectorstore = None

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create a new vector store from documents"""
        try:
            self.client.delete_collection(name=self.collection_name)
        except:
            pass
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            client=self.client
        )
        return self.vectorstore

    def get_vectorstore(self) -> Optional[Chroma]:
        """Get existing vector store or return None"""
        if self.vectorstore is None:
            try:
                collections = self.client.list_collections()
                collection_names = [col.name for col in collections]
                
                if self.collection_name in collection_names:
                    self.vectorstore = Chroma(
                        client=self.client,
                        collection_name=self.collection_name,
                        embedding_function=self.embeddings
                    )
                else:
                    return None
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return None
        
        return self.vectorstore

    def search(self, query: str, k: int = 4, filter_dict: Optional[dict] = None):
        """Search for similar documents"""
        if self.vectorstore is None:
            self.vectorstore = self.get_vectorstore()
        
        if self.vectorstore is None:
            return []
        
        if filter_dict:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        
        return results

    def delete_vectorstore(self):
        """Delete the vector store collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.vectorstore = None
        except Exception as e:
            print(f"Error deleting vector store: {e}")

    def list_collections(self):
        """List all available collections"""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []