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
FUSION_PROMPT = """你必须严格按照以下 JSON 格式返回结果。

任务：分析对话记录，提炼关键信息并为每条信息打上合适的标签

输出格式示例：
[
    {"content": "用户讨厌香菜", "tags": ["food_preference", "dislike_cilantro", "dietary"]},
    {"content": "用户喜欢辣味食物", "tags": ["food_preference", "spicy_lover", "taste"]},
    {"content": "简单的问候", "tags": ["social_greeting", "conversation_start"]},
    {"content": "《三体》中的黑暗森林法则", "tags": ["book_content", "sci_fi_concept", "theory", "three_body_problem"]}
]

字段说明：
- content: 简短的事实、观点或偏好描述，如果这句话中有时间信息，请保留下来
- tags: 标签数组，完全由你根据内容自主创建
  * 标签应该准确描述内容的特征、性质、来源、类型、情绪表达等
  * 标签可以是任何你认为合适的词汇，不受限制
  * 标签要有区分度，能帮助后续检索
  * 标签之间避免重复，如果意义相近，就不要多写一个，尽可能保证标签少
  * 标签应该是一些概括性的词汇，而不是内容的同义词
  * 标签的数量不定，不要超过5个
  * 如果这句话有强烈的喜好和情绪表达，请务必反映在标签中

规则：
1. 标签要有意义，有概括性，互相之间意义不要重复，而且标签中不要存在content的同义词汇
2. 将相关对话合并提炼，要求尽可能将内容表达完整
3. 标签用中文，content 用中文
4. 对于你觉得价值极低的话，比如打招呼什么的，请剔除掉
5. 标签中不要带上时间信息
6. 只返回 JSON 数组，不要有任何其他文字"""

# Prompt for consolidating similar memories
CONSOLIDATION_PROMPT = """你必须严格按照以下 JSON 格式返回结果。

任务：将以下相似的记忆合并成一条更完整、更简洁的记忆。

输出格式示例：
{
    "content": "用户2025年讨厌香菜，很多年前年前就是这样了",
    "tags": ["food_preference", "dislike_cilantro", "long_term_habit"]
}

要求：
1. 保留所有关键信息（地点、人物、事件），时间只保留最新的，其他的时间相对时间表示
2. 去除冗余和重复内容
3. 保持时间线的连贯性
4. 标签应该覆盖所有原始记忆的标签含义
5. content 字段要简洁但完整，不超过50字
6. 只返回 JSON 对象，不要有任何其他文字"""


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
                {"role": "user", "content": f"请分析以下对话记录：\n\n{raw_data_str}"}
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
    
    def consolidate_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Consolidate a group of similar memories into one.
        
        Args:
            memories: List of similar memory items to consolidate.
        
        Returns:
            Single consolidated memory with content and tags.
        """
        # Prepare memories for consolidation
        memories_str = json.dumps(memories, ensure_ascii=False, indent=2)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CONSOLIDATION_PROMPT},
                {"role": "user", "content": f"请合并以下相似的记忆：\n\n{memories_str}"}
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
            print(f"[MemoryPipeline] Consolidated {len(memories)} memories into 1")
            return consolidated
            
        except json.JSONDecodeError as e:
            print(f"[MemoryPipeline] Failed to parse consolidation JSON: {e}")
            print(f"[MemoryPipeline] Raw response: {content}")
            raise ValueError(f"Failed to parse consolidation response: {e}")
