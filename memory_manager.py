from typing import Dict, Any, List
from config import Settings
from storage.buffer import ShortTermBuffer
from storage.vector_db import VectorStorage
from processing.pipeline import MemoryPipeline


class AgentMemoryCore:
    """
    Main memory management system coordinating STM and LTM.
    Handles memory addition, recall, and automated flushing.
    """
    
    def __init__(self):
        """
        Initialize all memory components.
        """
        # Validate configuration
        Settings.validate()
        
        # Initialize components
        self.settings = Settings
        self.buffer = ShortTermBuffer()
        self.vector_storage = VectorStorage(db_path=Settings.DB_PATH)
        self.pipeline = MemoryPipeline(
            api_key=Settings.API_KEY,
            base_url=Settings.BASE_URL,
            model=Settings.MODEL
        )
        
        print(f"[AgentMemoryCore] Initialized with flush threshold: {Settings.FLUSH_THRESHOLD}")
    
    def add_memory(self, role: str, content: str) -> None:
        """
        Add a new memory item to the short-term buffer.
        Automatically triggers flush if threshold is reached.
        
        Args:
            role: The role of the speaker (e.g., 'user', 'assistant').
            content: The message content.
        """
        # Add to buffer
        self.buffer.add_item(role, content)
        print(f"[AgentMemoryCore] Added memory: {role} - {content[:50]}...")
        
        # Check if flush is needed
        if self.buffer.count() >= self.settings.FLUSH_THRESHOLD:
            print(f"[AgentMemoryCore] Flush threshold reached ({self.buffer.count()}/{self.settings.FLUSH_THRESHOLD})")
            self._flush()
    
    def recall(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Recall relevant memories based on a query.
        Returns both short-term (recent) and long-term (relevant) context.
        
        Args:
            query: The query string to search for.
            top_k: Number of top results to return from LTM.
        
        Returns:
            Dictionary containing STM and LTM contexts.
        """
        # Get recent context from short-term memory
        stm_context = self.buffer.get_all()
        
        # Search related context from long-term memory (active collection)
        ltm_context = self.vector_storage.search_active(query, top_k=top_k)
        
        print(f"[AgentMemoryCore] Recall: {len(stm_context)} STM items, {len(ltm_context)} LTM items")
        
        return {
            "short_term": stm_context,
            "long_term": ltm_context
        }
    
    def _flush(self) -> None:
        """
        Internal method to flush STM to LTM.
        Processes memories through the pipeline and stores them based on score.
        """
        print("[AgentMemoryCore] Starting flush process...")
        
        # Get all data from buffer
        raw_memories = self.buffer.get_all()
        
        if not raw_memories:
            print("[AgentMemoryCore] No memories to flush")
            return
        
        # Process batch through pipeline
        print(f"[AgentMemoryCore] Processing {len(raw_memories)} memories...")
        processed_memories = self.pipeline.process_batch(raw_memories)
        
        # Store all processed memories with tags
        stored_count = 0
        
        for item in processed_memories:
            content = item.get("content", "")
            tags = item.get("tags", [])
            
            # Combine content with tags for better embedding
            tags_str = ", ".join(tags)
            content_with_tags = f"{content} [标签: {tags_str}]"
            
            metadata = {
                "tags": tags_str,
                "original_content": content,  # Store original content without tags
                "original_count": len(raw_memories)
            }
            
            # Store all memories to active collection regardless of tags
            self.vector_storage.add_active(content_with_tags, metadata)
            stored_count += 1
            print(f"[AgentMemoryCore] → Stored: [{tags_str}] {content[:60]}...")
        
        # Clear buffer
        self.buffer.clear()
        
        print(f"[AgentMemoryCore] Flush complete: {stored_count} memories stored")
    
    def consolidate_memories(self, similarity_threshold: float = 0.85) -> int:
        """
        Consolidate similar memories in LTM by grouping and merging them.
        
        Args:
            similarity_threshold: Cosine similarity threshold (0-1) for grouping.
                                Higher = stricter grouping. Default 0.85.
        
        Returns:
            Number of memories after consolidation.
        """
        print(f"[AgentMemoryCore] Starting memory consolidation (threshold={similarity_threshold})...")
        
        # Get all active memories
        all_memories = self.vector_storage.get_all_active()
        
        if len(all_memories) < 2:
            print(f"[AgentMemoryCore] Not enough memories to consolidate ({len(all_memories)} total)")
            return len(all_memories)
        
        print(f"[AgentMemoryCore] Retrieved {len(all_memories)} memories from LTM")
        
        # Calculate cosine similarity matrix
        import numpy as np
        embeddings = np.array([m['embedding'] for m in all_memories])
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        
        # Compute similarity matrix
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Group similar memories using greedy clustering
        visited = set()
        groups = []
        
        for i in range(len(all_memories)):
            if i in visited:
                continue
            
            group = [i]
            visited.add(i)
            
            for j in range(i + 1, len(all_memories)):
                if j in visited:
                    continue
                
                if similarity_matrix[i][j] >= similarity_threshold:
                    group.append(j)
                    visited.add(j)
            
            groups.append(group)
        
        print(f"[AgentMemoryCore] Grouped into {len(groups)} clusters")
        
        # Consolidate each group
        consolidated_count = 0
        memories_to_delete = []
        
        for group_indices in groups:
            if len(group_indices) <= 1:
                # No need to consolidate single memories
                continue
            
            # Get memories in this group
            group_memories = [{
                "content": all_memories[i]['metadata'].get('original_content', all_memories[i]['document']),
                "tags": all_memories[i]['metadata'].get('tags', '').split(', ')
            } for i in group_indices]
            
            # Consolidate using LLM
            try:
                consolidated = self.pipeline.consolidate_memories(group_memories)
                
                # Create new consolidated memory
                tags_str = ", ".join(consolidated['tags'])
                content_with_tags = f"{consolidated['content']} [标签: {tags_str}]"
                
                metadata = {
                    "tags": tags_str,
                    "original_content": consolidated['content'],
                    "consolidated_from": len(group_indices),
                    "original_count": sum(all_memories[i]['metadata'].get('original_count', 1) for i in group_indices)
                }
                
                # Store consolidated memory
                self.vector_storage.add_active(content_with_tags, metadata)
                
                # Mark old memories for deletion
                memories_to_delete.extend([all_memories[i]['id'] for i in group_indices])
                
                consolidated_count += 1
                print(f"[AgentMemoryCore] ✓ Consolidated {len(group_indices)} memories: {consolidated['content'][:60]}...")
                
            except Exception as e:
                print(f"[AgentMemoryCore] ✗ Failed to consolidate group: {e}")
        
        # Delete old memories
        if memories_to_delete:
            self.vector_storage.delete_memories(memories_to_delete)
        
        final_count = len(all_memories) - len(memories_to_delete) + consolidated_count
        print(f"[AgentMemoryCore] Consolidation complete: {len(all_memories)} → {final_count} memories")
        
        return final_count
