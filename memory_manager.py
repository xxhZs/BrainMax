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
    
    def recall(self, query: str, top_k: int = 5, include_cold: bool = True) -> Dict[str, Any]:
        """
        Recall relevant memories based on a query.
        Returns short-term, active long-term, and cold storage context.
        
        Args:
            query: The query string to search for.
            top_k: Number of top results to return from each collection.
            include_cold: Whether to include cold storage in the search.
        
        Returns:
            Dictionary containing STM, active LTM, and cold storage contexts.
        """
        # Get recent context from short-term memory
        stm_context = self.buffer.get_all()
        
        # Search related context from long-term memory (active collection)
        active_context = self.vector_storage.search_active(query, top_k=top_k)
        
        # Search cold storage if enabled
        cold_context = []
        if include_cold:
            cold_context = self.vector_storage.search_cold(query, top_k=top_k)
        
        print(f"[AgentMemoryCore] Recall: {len(stm_context)} STM items, "
              f"{len(active_context)} active items, {len(cold_context)} cold items")
        
        return {
            "short_term": stm_context,
            "active_long_term": active_context,
            "cold_storage": cold_context
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
            
            # Store only content for embedding (without tags)
            tags_str = ", ".join(tags)
            
            metadata = {
                "tags": tags_str,
                "original_content": content,  # Store original content without tags
                "original_count": len(raw_memories)
            }
            
            # Store memories with pure content for better semantic matching
            self.vector_storage.add_active(content, metadata)
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
        memories_to_cold = []
        
        for group_indices in groups:
            if len(group_indices) <= 1:
                # No need to consolidate single memories
                continue
            
            # Collect memories to move to cold storage
            group_cold_memories = [all_memories[i] for i in group_indices]
            
            # Move to cold storage and get the cold IDs
            cold_ids = self.vector_storage.move_to_cold(group_cold_memories)
            
            # Get memories in this group
            group_memories = [{
                "content": all_memories[i]['metadata'].get('original_content', all_memories[i]['document']),
                "tags": all_memories[i]['metadata'].get('tags', '').split(', ')
            } for i in group_indices]
            
            # Consolidate using LLM (may return multiple groups)
            try:
                consolidated_list = self.pipeline.consolidate_memories(group_memories)
                
                # Process each consolidated memory
                for consolidated in consolidated_list:
                    tags_str = ", ".join(consolidated['tags'])
                    
                    metadata = {
                        "tags": tags_str,
                        "original_content": consolidated['content'],
                        "consolidated_from": len(group_indices),
                        "original_count": sum(all_memories[i]['metadata'].get('original_count', 1) for i in group_indices),
                        "cold_ids": ",".join(cold_ids)  # Pointer to cold storage
                    }
                    
                    # Store consolidated memory (without tags in content)
                    self.vector_storage.add_active(consolidated['content'], metadata)
                    consolidated_count += 1
                    print(f"[AgentMemoryCore] ✓ Consolidated memory: {consolidated['content'][:60]}...")
                
                # Collect for tracking
                memories_to_cold.extend(group_cold_memories)
                
                # Mark old memories for deletion
                memories_to_delete.extend([all_memories[i]['id'] for i in group_indices])
                for i in group_indices:
                    memories_to_cold.append(all_memories[i])
                
                # Mark old memories for deletion
                memories_to_delete.extend([all_memories[i]['id'] for i in group_indices])
                
            except Exception as e:
                print(f"[AgentMemoryCore] ✗ Failed to consolidate group: {e}")
        
        # Delete old memories from active collection (remove duplicates)
        if memories_to_delete:
            unique_ids = list(set(memories_to_delete))
            self.vector_storage.delete_memories(unique_ids)
            print(f"[AgentMemoryCore] Deleted {len(unique_ids)} unique memories (from {len(memories_to_delete)} total)")
        
        final_count = len(all_memories) - len(set(memories_to_delete)) + consolidated_count
        print(f"[AgentMemoryCore] Consolidation complete: {len(all_memories)} → {final_count} memories")
        print(f"[AgentMemoryCore] Moved {len(memories_to_cold)} memories to cold storage")
        
        return final_count
    
    def search_cold_memories(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search cold storage for relevant memories.
        
        Args:
            query: The query string to search for.
            top_k: Number of top results to return.
        
        Returns:
            Dictionary containing cold memory search results.
        """
        cold_context = self.vector_storage.search_cold(query, top_k=top_k)
        print(f"[AgentMemoryCore] Found {len(cold_context)} cold memories")
        
        return {
            "cold_memories": cold_context
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all memory collections.
        
        Returns:
            Dictionary with memory counts across all storage types.
        """
        stats = {
            "short_term": self.buffer.count(),
            "active_long_term": self.vector_storage.active_collection.count(),
            "cold_storage": self.vector_storage.cold_collection.count()
        }
        
        print(f"[AgentMemoryCore] Memory stats: STM={stats['short_term']}, "
              f"Active={stats['active_long_term']}, Cold={stats['cold_storage']}")
        
        return stats
    
    def get_cold_source(self, active_memory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get the cold storage source memories for an active memory.
        
        Args:
            active_memory: An active memory dict with metadata containing cold_ids.
        
        Returns:
            List of cold memory items that were consolidated into this active memory.
        """
        cold_ids_str = active_memory.get('metadata', {}).get('cold_ids', '')
        if not cold_ids_str:
            print(f"[AgentMemoryCore] No cold storage reference found")
            return []
        
        cold_ids = cold_ids_str.split(',')
        cold_memories = self.vector_storage.get_cold_memories(cold_ids)
        
        print(f"[AgentMemoryCore] Retrieved {len(cold_memories)} cold source memories")
        return cold_memories
    
    def smart_recall(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Intelligent recall that first retrieves hot data, then uses LLM to decide
        if detailed cold data is needed.
        
        Args:
            query: The query string to search for.
            top_k: Number of top results to return from active memory.
        
        Returns:
            Dictionary containing hot data and conditionally loaded cold data.
        """
        print(f"[AgentMemoryCore] Smart recall for query: {query[:60]}...")
        
        # Step 1: Get STM and hot data only
        stm_context = self.buffer.get_all()
        active_context = self.vector_storage.search_active(query, top_k=top_k)
        
        print(f"[AgentMemoryCore] Retrieved {len(stm_context)} STM items, {len(active_context)} active items")
        
        # If no active memories, return early
        if not active_context:
            return {
                "short_term": stm_context,
                "active_long_term": active_context,
                "cold_storage": [],
                "cold_needed": False
            }
        
        # Step 2: Ask LLM if more detailed data is needed
        needs_detail = self._check_if_detail_needed(query, active_context)
        
        # Step 3: If needed, retrieve cold data using hot memory IDs
        cold_data = []
        hot_to_cold_mapping = {}
        
        if needs_detail['needed'] and needs_detail['hot_memory_ids']:
            print(f"[AgentMemoryCore] LLM requested details for hot IDs: {needs_detail['hot_memory_ids']}")
            
            for hot_id in needs_detail['hot_memory_ids']:
                # Find the active memory by ID
                active_mem = None
                for mem in active_context:
                    if mem['id'] == hot_id:
                        active_mem = mem
                        break
                
                if active_mem:
                    # Get cold data using the pointer
                    cold_sources = self.get_cold_source(active_mem)
                    cold_data.extend(cold_sources)
                    hot_to_cold_mapping[hot_id] = [c['id'] for c in cold_sources]
                    print(f"[AgentMemoryCore] Hot ID {hot_id} → {len(cold_sources)} cold items")
        
        print(f"[AgentMemoryCore] Smart recall complete: {len(cold_data)} cold items loaded")
        
        return {
            "short_term": stm_context,
            "active_long_term": active_context,
            "cold_storage": cold_data,
            "cold_needed": needs_detail['needed'],
            "detail_reason": needs_detail.get('reason', ''),
            "hot_to_cold_mapping": hot_to_cold_mapping  # Show which hot ID maps to which cold IDs
        }
    
    def _check_if_detail_needed(self, query: str, active_memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use LLM to determine if detailed cold data is needed.
        
        Args:
            query: The user's query.
            active_memories: List of active memory items with IDs.
        
        Returns:
            Dict with 'needed' (bool), 'hot_memory_ids' (list of IDs), and 'reason' (str).
        """
        # Format active memories for LLM with their IDs
        memories_text = []
        id_mapping = {}  # Map index to memory ID
        
        for i, mem in enumerate(active_memories):
            mem_id = mem['id']
            content = mem['metadata'].get('original_content', mem['document'])
            id_mapping[i] = mem_id
            memories_text.append(f"[Index {i}, ID: {mem_id}] {content}")
        
        prompt = f"""You are a memory system analyzer. A user has asked: "{query}"

The system retrieved the following summarized/consolidated hot memories:

{chr(10).join(memories_text)}

Task: Determine if these consolidated memories are sufficient to answer the query, or if more detailed original cold data is needed.

Respond in JSON format:
{{
    "needed": true/false,
    "memory_indices": [0, 1, 2],  // indices of memories that need cold details (empty if not needed)
    "reason": "explanation why details are/aren't needed"
}}

Guidelines:
- If consolidated hot memories contain enough information, set needed=false
- If query requires specific details, examples, or nuances that might be lost in consolidation, set needed=true
- Only request details for memories that are relevant and likely to contain needed information
- Use the Index numbers to specify which memories need details"""

        try:
            response = self.pipeline.client.chat.completions.create(
                model=self.pipeline.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            import json
            # Extract JSON if wrapped in markdown
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(result_text)
            
            # Convert indices to actual memory IDs
            indices = result.get("memory_indices", [])
            hot_memory_ids = [id_mapping[idx] for idx in indices if idx in id_mapping]
            
            print(f"[AgentMemoryCore] LLM decision: needed={result['needed']}, "
                  f"hot_ids={hot_memory_ids}, reason={result['reason'][:60]}...")
            
            return {
                "needed": result.get("needed", False),
                "hot_memory_ids": hot_memory_ids,
                "reason": result.get("reason", "")
            }
            
        except Exception as e:
            print(f"[AgentMemoryCore] Error in LLM detail check: {e}")
            # Default to not needing details on error
            return {
                "needed": False,
                "hot_memory_ids": [],
                "reason": "Error in analysis, using hot data only"
            }
