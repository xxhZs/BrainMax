import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """
    Configuration settings loaded from environment variables.
    """
    
    # OpenAI API Configuration (for LLM)
    API_KEY = os.getenv("OPENAI_API_KEY", "")
    BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Embedding API Configuration (for vector search)
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-3")
    EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
    
    # Chroma Vector Database Configuration
    DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    # Memory Management Configuration
    FLUSH_THRESHOLD = int(os.getenv("FLUSH_THRESHOLD", "5"))
    
    @classmethod
    def validate(cls):
        """
        Validate required configuration settings.
        Raises ValueError if required settings are missing.
        """
        if not cls.API_KEY:
            raise ValueError("OPENAI_API_KEY is required but not set")
        if not cls.EMBEDDING_API_KEY:
            raise ValueError("EMBEDDING_API_KEY is required but not set")
