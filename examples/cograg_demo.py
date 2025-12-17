#!/usr/bin/env python3
"""Cog-RAG Demo Script.

This example demonstrates how to use Cog-RAG for knowledge retrieval
with dual-hypergraph architecture and theme alignment.

Usage:
    python examples/cograg_demo.py

Requirements:
    - Configure my_config.py with your LLM and embedding API settings
    - Place your text data in examples/mock_data.txt
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import logging
from typing import List


from cograg import CogRAG, QueryParam

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import LLM and embedding functions
from cograg.utils import EmbeddingFunc
from cograg.llm import openai_embedding, openai_complete_if_cache

# Import configuration (create my_config.py from config_temp.py)
try:
    from my_config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    from my_config import EMB_API_KEY, EMB_BASE_URL, EMB_MODEL, EMB_DIM
except ImportError:
    logger.error("Please create my_config.py from config_temp.py with your API settings")
    raise


async def llm_model_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[dict] = None,
    **kwargs,
) -> str:
    """Custom LLM function using OpenAI-compatible API.
    
    Args:
        prompt: The user prompt to send.
        system_prompt: Optional system message.
        history_messages: Optional conversation history.
        **kwargs: Additional arguments for the API.
        
    Returns:
        Generated response text.
    """
    if history_messages is None:
        history_messages = []
        
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        **kwargs,
    )


async def embedding_func(texts: List[str]) -> np.ndarray:
    """Custom embedding function using OpenAI-compatible API.
    
    Args:
        texts: List of text strings to embed.
        
    Returns:
        NumPy array of embeddings.
    """
    return await openai_embedding(
        texts,
        model=EMB_MODEL,
        api_key=EMB_API_KEY,
        base_url=EMB_BASE_URL,
    )


def insert_texts_with_retry(
    rag: CogRAG,
    texts: str,
    retries: int = 3,
    delay: int = 5,
) -> None:
    """Insert texts into RAG with automatic retry on failure.
    
    Args:
        rag: CogRAG instance.
        texts: Text content to insert.
        retries: Maximum number of retry attempts.
        delay: Seconds to wait between retries.
        
    Raises:
        RuntimeError: If all retry attempts fail.
    """
    for attempt in range(retries):
        try:
            rag.insert(texts)
            logger.info("Successfully inserted texts into RAG")
            return
        except Exception as e:
            logger.warning(
                f"Insertion failed (attempt {attempt + 1}/{retries}): {e}. "
                f"Retrying in {delay} seconds..."
            )
            time.sleep(delay)
            
    raise RuntimeError("Failed to insert texts after multiple retries.")


def main():
    """Main demo function."""
    # Setup working directory
    data_name = "mock"
    WORKING_DIR = Path("caches") / data_name
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Initializing CogRAG with working directory: {WORKING_DIR}")
    
    # Initialize CogRAG
    rag = CogRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMB_DIM,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    # Load and insert data
    mock_data_file_path = Path("./examples/mock_data.txt")
    if not mock_data_file_path.exists():
        logger.error(f"Data file not found: {mock_data_file_path}")
        return
        
    with open(mock_data_file_path, "r", encoding="utf-8") as file:
        texts = file.read()

    logger.info(f"Inserting {len(texts)} characters of text...")
    insert_texts_with_retry(rag, texts)

    # Perform different types of queries and handle potential errors

    try:
        print("\n\n\nPerforming Cog-RAG...")
        print(
            rag.query(
                "What are the top themes in this story?", 
                param=QueryParam(mode="cog")
            )
        )
    except Exception as e:
        logger.error(f"Cog-RAG query failed: {e}")

    try:
        print("\n\n\nPerforming Naive RAG...")
        print(
            rag.query(
                "What are the top themes in this story?",
                param=QueryParam(mode="naive")
            )
        )
    except Exception as e:
        logger.error(f"Naive RAG query failed: {e}")


if __name__ == "__main__":
    main()
