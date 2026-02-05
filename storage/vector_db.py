import chromadb
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from openai import OpenAI
from config import Settings


class ZhipuEmbeddingFunction:
    """
    Custom embedding function for ZhipuAI's embedding-3 model.
    """
    def __init__(self, api_key: str, model: str, base_url: str, dimensions: int = 1024):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.dimensions = dimensions
    
    def name(self) -> str:
        """Return the name of the embedding function."""
        return f"zhipu-{self.model}"
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for input texts (batch).
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=input,
            dimensions=self.dimensions
        )
        return [item.embedding for item in response.data]
    
    def embed_query(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for query texts.
        ChromaDB's query() expects this method to return List[List[float]].
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=input,
                dimensions=self.dimensions
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"[ZhipuEmbedding] Error in embed_query: {e}")
            print(f"[ZhipuEmbedding] Input: {input}")
            raise


class VectorStorage:
    """
    Vector storage using ChromaDB with persistent storage.
    Maintains two separate collections for active and archived memories.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize ChromaDB client and create collections.
        
        Args:
            db_path: Path to the persistent ChromaDB storage directory.
        """
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Initialize ZhipuAI embedding function
        self.embedding_function = ZhipuEmbeddingFunction(
            api_key=Settings.EMBEDDING_API_KEY,
            model=Settings.EMBEDDING_MODEL,
            base_url=Settings.EMBEDDING_BASE_URL,
            dimensions=Settings.EMBEDDING_DIMENSIONS
        )
        print(f"[VectorStorage] Using {Settings.EMBEDDING_MODEL} ({Settings.EMBEDDING_DIMENSIONS}D) from {Settings.EMBEDDING_BASE_URL}")
        
        # Collection for fused insights and high-score facts
        self.active_collection = self.client.get_or_create_collection(
            name="active_memory",
            embedding_function=self.embedding_function,
            metadata={"description": "Active memory for fused insights and important facts"}
        )
        
        # Collection for raw logs and low-score items (forgotten memories)
        self.archive_collection = self.client.get_or_create_collection(
            name="archive_memory",
            embedding_function=self.embedding_function,
            metadata={"description": "Archived memory for raw logs and low-priority items"}
        )
        
        print(f"[VectorStorage] Initialized ChromaDB with {self.active_collection.count()} active and {self.archive_collection.count()} archived memories")
    
    def add_active(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a memory item to the active collection.
        
        Args:
            text: The memory content to store.
            metadata: Optional metadata dictionary.
        
        Returns:
            The generated ID for the stored item.
        """
        # Generate unique ID based on content hash + timestamp
        content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        item_id = f"active_{content_hash}_{datetime.now().timestamp()}"
        
        # Add metadata if not provided
        if metadata is None:
            metadata = {}
        metadata["added_at"] = datetime.now().isoformat()
        
        self.active_collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[item_id]
        )
        
        return item_id
    
    def add_archive(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a memory item to the archive collection.
        
        Args:
            text: The memory content to store.
            metadata: Optional metadata dictionary.
        
        Returns:
            The generated ID for the stored item.
        """
        # Generate unique ID based on content hash + timestamp
        content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        item_id = f"archive_{content_hash}_{datetime.now().timestamp()}"
        
        # Add metadata if not provided
        if metadata is None:
            metadata = {}
        metadata["archived_at"] = datetime.now().isoformat()
        
        self.archive_collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[item_id]
        )
        
        return item_id
    
    def search_active(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the active memory collection for relevant memories.
        Uses ChromaDB's vector similarity search.
        
        Args:
            query: The search query string.
            top_k: Number of top results to return.
        
        Returns:
            List of dictionaries containing documents and metadata.
        """
        results = self.active_collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if 'distances' in results else None
                })
        
        return formatted_results
    
    def get_all_active(self) -> List[Dict[str, Any]]:
        """
        Get all memories from the active collection.
        
        Returns:
            List of all active memory items.
        """
        result = self.active_collection.get(
            include=["documents", "metadatas", "embeddings"]
        )
        
        memories = []
        if result['ids']:
            for i in range(len(result['ids'])):
                memories.append({
                    "id": result['ids'][i],
                    "document": result['documents'][i],
                    "metadata": result['metadatas'][i],
                    "embedding": result['embeddings'][i]
                })
        
        return memories
    
    def delete_memories(self, ids: List[str]) -> None:
        """
        Delete memories by IDs from active collection.
        
        Args:
            ids: List of memory IDs to delete.
        """
        if ids:
            self.active_collection.delete(ids=ids)
            print(f"[VectorStorage] Deleted {len(ids)} memories")


