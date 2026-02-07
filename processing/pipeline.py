import json
from typing import List, Dict, Any
from openai import OpenAI
from pydantic import BaseModel


# Pydantic models for structured output
class MemoryItem(BaseModel):
    content: str
    tags: List[str]


class MemoryFusionResult(BaseModel):
    memories: List[MemoryItem]


# Prompt for memory fusion and tagging
FUSION_PROMPT = """You must strictly return results in the following JSON format.

Task: Analyze conversation records, extract key information and add appropriate tags for each piece. Output multiple memory entries - don't over-merge different information into one.

Output format example:
[
    {"content": "User dislikes cilantro", "tags": ["food_preference", "dislike_cilantro", "dietary"]},
    {"content": "User enjoys spicy food", "tags": ["food_preference", "spicy_lover", "taste"]},
    {"content": "User mentioned liking Thai cuisine", "tags": ["food_preference", "cuisine_type", "asian_food"]},
    {"content": "Dark Forest theory from Three Body Problem", "tags": ["book_content", "sci_fi_concept", "theory", "three_body_problem"]}
]

Field descriptions:
- content: Brief description of facts, opinions or preferences. If there is time information, preserve it. Be as complete as possible.
- tags: Tag array specific to THIS content entry, created completely autonomously by you
  * Each content entry must have its own independent tags
  * Tags should accurately describe content characteristics, nature, source, type, emotional expression, etc.
  * Tags can be any words you consider appropriate, without restrictions
  * Tags should be distinctive to help with retrieval
  * Avoid duplicate tags within the same entry; if meanings are similar, don't add another one
  * Number of tags is flexible per entry, but don't exceed 5 per entry
  * If the statement has strong preferences or emotional expression, must reflect in tags

Rules:
1. Output multiple content entries - there is no upper limit on the number of entries
2. Each entry should cover ONE piece of information with its own tags
3. Don't over-merge different topics into one entry - keep them separate
4. Tags must be meaningful, general, non-repetitive in meaning, and must not contain synonyms of content
5. Merge and refine only closely related dialogues about the SAME topic
6. Use English for both tags and content
7. For statements you consider extremely low value (like greetings), please remove them
8. Do not include time information in tags
9. Return only the JSON array, no other text"""

# Prompt for consolidating similar memories
CONSOLIDATION_PROMPT = """You must strictly return results in the following JSON format.

Task: Analyze the following similar memories and merge them intelligently. You can merge them into one, or multiple groups based on their semantic similarity.

Output format example:
[
    {
        "content": "User has disliked cilantro since 2025, this preference existed many years before",
        "tags": ["food_preference", "dislike_cilantro", "long_term_habit"]
    },
    {
        "content": "User enjoys spicy food and hot sauces",
        "tags": ["food_preference", "spicy_lover", "taste"]
    }
]

Requirements:
1. Group memories by semantic similarity - if they talk about different topics, keep them separate
2. For each group, preserve all key information (location, people, events). Keep only the latest time, express other times relatively
3. Remove redundancy and duplicate content within each group
4. Maintain timeline coherence
5. Tags should merge and cover all tags from grouped memories, maximum 10 tags per group. Remove duplicates and similar meanings.
6. Do not merge content from different subjects.
7. Return only the JSON array, no other text"""


class MemoryPipeline:
    """
    Memory processing pipeline using LLM for fusion and scoring.
    Processes raw memory items into refined insights with importance scores.
    """
    
    def __init__(self, api_key: str, base_url: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the pipeline with OpenAI client.
        
        Args:
            api_key: OpenAI API key.
            base_url: Base URL for the OpenAI API.
            model: Model name to use for completions.
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
    
    def process_batch(self, raw_memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of raw memories using LLM.
        
        Args:
            raw_memories: List of raw memory dictionaries.
        
        Returns:
            List of processed memory items with content and tags.
        """
        # Convert raw memories to JSON string
        raw_data_str = json.dumps(raw_memories, ensure_ascii=False, indent=2)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": FUSION_PROMPT},
                {"role": "user", "content": f"Please analyze the following conversation records:\n\n{raw_data_str}"}
            ],
            temperature=0.3
        )
        
        # Extract JSON from response
        content = response.choices[0].message.content
        
        # Try to parse JSON
        try:
            # Remove markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            memories = json.loads(content)
            
            # Ensure it's a list
            if isinstance(memories, dict) and "memories" in memories:
                memories = memories["memories"]
            
            print(f"[MemoryPipeline] Successfully extracted {len(memories)} memories")
            return memories
            
        except json.JSONDecodeError as e:
            print(f"[MemoryPipeline] Failed to parse JSON: {e}")
            print(f"[MemoryPipeline] Raw response: {content}")
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
    
    def consolidate_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Consolidate a group of similar memories into one or more consolidated memories.
        LLM decides how many groups to create based on semantic similarity.
        
        Args:
            memories: List of similar memory items to consolidate.
        
        Returns:
            List of consolidated memories with content and tags.
        """
        # Prepare memories for consolidation
        memories_str = json.dumps(memories, ensure_ascii=False, indent=2)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CONSOLIDATION_PROMPT},
                {"role": "user", "content": f"Please analyze and merge the following similar memories:\n\n{memories_str}"}
            ],
            temperature=0.3
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            consolidated = json.loads(content)
            
            # Ensure it's a list
            if isinstance(consolidated, dict):
                consolidated = [consolidated]
            
            print(f"[MemoryPipeline] Consolidated {len(memories)} memories into {len(consolidated)} groups")
            return consolidated
            
        except json.JSONDecodeError as e:
            print(f"[MemoryPipeline] Failed to parse consolidation JSON: {e}")
            print(f"[MemoryPipeline] Raw response: {content}")
            raise ValueError(f"Failed to parse consolidation response: {e}")
