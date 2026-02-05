from datetime import datetime
from typing import List, Dict, Any


class ShortTermBuffer:
    """
    Short-term memory buffer using a simple list.
    Stores recent conversation items before they are flushed to long-term memory.
    """
    
    def __init__(self):
        """
        Initialize an empty buffer.
        """
        self._buffer: List[Dict[str, Any]] = []
    
    def add_item(self, role: str, content: str) -> None:
        """
        Add a new item to the buffer.
        
        Args:
            role: The role of the speaker (e.g., 'user', 'assistant').
            content: The message content.
        """
        item = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self._buffer.append(item)
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all items in the buffer.
        
        Returns:
            List of all memory items.
        """
        return self._buffer
    
    def clear(self) -> None:
        """
        Clear all items from the buffer.
        """
        self._buffer = []
    
    def count(self) -> int:
        """
        Get the current number of items in the buffer.
        
        Returns:
            Number of items in the buffer.
        """
        return len(self._buffer)
