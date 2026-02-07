# BrainMax

🧠 **An AI Agent Memory Management System** with multi-tier storage, vector-based semantic search, and LLM-powered memory consolidation.

> 🚧 **Work in Progress** — Core functionality is implemented and available for testing.

## Features

- **Multi-tier Memory Architecture** — Short-term memory (STM), active long-term memory (LTM), and cold storage work together to balance recall speed and capacity.
- **Vector-based Semantic Search** — Uses embedding models to perform similarity matching, enabling intelligent memory recall beyond keyword matching.
- **Automated Memory Consolidation** — Groups similar memories using cosine similarity and merges them via LLM analysis, keeping the memory store lean and relevant.
- **Smart Recall** — LLM-powered decision-making that loads cold storage details only when the consolidated hot data is insufficient to answer a query.
- **Tag-based Organization** — Automatically tags memories during the flush pipeline for better categorization and filtering.
- **Configurable LLM & Embedding Backends** — Supports OpenAI-compatible APIs (e.g., DeepSeek) for LLM and Zhipu AI for embeddings.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AgentMemoryCore                         │
│                    (memory_manager.py)                       │
├──────────┬──────────────────────────┬───────────────────────┤
│   STM    │       Active LTM         │     Cold Storage      │
│ (Buffer) │    (ChromaDB active)     │  (ChromaDB cold)      │
│          │                          │                       │
│ In-mem   │  Consolidated memories   │  Original memories    │
│ FIFO     │  with tags & embeddings  │  before consolidation │
└──────────┴──────────────────────────┴───────────────────────┘
         │                                     ▲
         ▼                                     │
   MemoryPipeline  ───── flush & consolidate ──┘
   (processing/pipeline.py)
```

**Memory Flow:**

1. New memories are added to the **Short-Term Buffer** (in-memory FIFO list).
2. When the buffer reaches the flush threshold, the **MemoryPipeline** processes and tags the batch using the LLM.
3. Processed memories are stored in **Active LTM** (ChromaDB `active` collection).
4. During consolidation, similar active memories are grouped, merged via LLM, and originals are moved to **Cold Storage** (`cold` collection).
5. On recall, **Smart Recall** retrieves hot data first and uses the LLM to decide whether to load cold details.

## Project Structure

```
BrainMax/
├── config.py              # Configuration via environment variables
├── main.py                # Demo / test script
├── memory_manager.py      # AgentMemoryCore — main orchestrator
├── requirements.txt       # Python dependencies
├── processing/
│   ├── __init__.py
│   └── pipeline.py        # MemoryPipeline: LLM processing & consolidation
├── storage/
│   ├── __init__.py
│   ├── buffer.py          # ShortTermBuffer: in-memory recent context
│   └── vector_db.py       # VectorStorage: ChromaDB wrapper with custom embeddings
└── LICENSE                # Apache 2.0
```

## Getting Started

### Prerequisites

- Python 3.8+
- An OpenAI-compatible API key (e.g., DeepSeek)
- An embedding API key (e.g., Zhipu AI)

### Installation

```bash
git clone https://github.com/xxhZs/BrainMax.git
cd BrainMax
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
# LLM Configuration (OpenAI-compatible)
OPENAI_API_KEY=your_llm_api_key
OPENAI_BASE_URL=https://api.deepseek.com        # default
OPENAI_MODEL=gpt-3.5-turbo                      # default

# Embedding Configuration (Zhipu AI)
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4  # default
EMBEDDING_MODEL=embedding-3                      # default
EMBEDDING_DIMENSIONS=1024                        # default

# Storage
CHROMA_DB_PATH=./chroma_db                       # default

# Memory Management
FLUSH_THRESHOLD=1000                             # default, items before STM→LTM flush
```

Only `OPENAI_API_KEY` and `EMBEDDING_API_KEY` are required. All other values have sensible defaults.

### Usage

Run the included demo script:

```bash
python main.py
```

Or use `AgentMemoryCore` in your own code:

```python
from memory_manager import AgentMemoryCore

core = AgentMemoryCore()

# Add memories
core.add_memory("user", "I hate coriander, remember that.")
core.add_memory("assistant", "Got it! I'll remember you don't like coriander.")

# Recall relevant memories
result = core.recall("What food do I dislike?", top_k=5)
print(result["short_term"])       # Recent context
print(result["active_long_term"]) # Relevant long-term memories

# Consolidate similar memories
core.consolidate_memories(similarity_threshold=0.7)

# Smart recall — loads cold details only when needed
result = core.smart_recall("Tell me about my food preferences")

# Get memory statistics
stats = core.get_memory_stats()
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `openai` ≥ 0.27.0 | LLM interaction (OpenAI-compatible APIs) |
| `chromadb` ≥ 0.3.21 | Vector database for persistent memory storage |
| `python-dotenv` ≥ 1.0.0 | Load environment variables from `.env` |

NumPy and Pydantic are pulled in as transitive dependencies.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
