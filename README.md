<div align="center">

# 🧠 Cog-RAG

**Cognitive-Inspired Dual-Hypergraph with Theme Alignment Retrieval-Augmented Generation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[English](#overview) | [中文](#概述)

</div>

---

## Overview

**Cog-RAG** is a cognitive-inspired retrieval-augmented generation framework that utilizes dual-hypergraph structures with theme alignment for enhanced knowledge retrieval and question answering. Unlike traditional RAG systems that rely on simple vector similarity search, Cog-RAG models complex multi-entity relationships through hypergraphs and implements a two-stage theme-entity alignment mechanism inspired by human cognitive processes.

### ✨ Key Features

- 🔗 **Dual-Hypergraph Architecture**: Separates entity-level and theme-level knowledge representation
- 🎯 **High-Order Relationship Modeling**: Captures multi-entity relationships beyond binary edges
- 🧠 **Cognitive-Inspired Two-Stage Retrieval**: Theme alignment followed by entity alignment
- 🔄 **Multiple Query Modes**: Supports various retrieval strategies for different use cases
- 📦 **Easy Integration**: Simple API design compatible with various LLM providers

### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Cog-RAG System                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Documents  │───▶│   Chunking  │───▶│  Entity Extraction  │  │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘  │
│                                                   │              │
│         ┌─────────────────────────────────────────┴───┐          │
│         ▼                                             ▼          │
│  ┌──────────────────┐                    ┌──────────────────┐   │
│  │  Entity-Relation │                    │   Key-Theme      │   │
│  │   Hypergraph     │                    │   Hypergraph     │   │
│  │                  │                    │                  │   │
│  │  ┌───┐   ┌───┐   │                    │  ┌───┐   ┌───┐   │   │
│  │  │ E │───│ E │   │                    │  │ K │───│ K │   │   │
│  │  └─┬─┘   └─┬─┘   │                    │  └─┬─┘   └─┬─┘   │   │
│  │    │   ╲   │     │                    │    │   ╲   │     │   │
│  │    │    ╲  │     │                    │    │    ╲  │     │   │
│  │  ┌─┴─┐  ┌┴──┐    │                    │  ┌─┴─┐  ┌┴──┐    │   │
│  │  │ E │──│ E │    │                    │  │ K │──│ K │    │   │
│  │  └───┘  └───┘    │                    │  └───┘  └───┘    │   │
│  └──────────────────┘                    └──────────────────┘   │
│           │                                       │              │
│           └──────────────────┬────────────────────┘              │
│                              ▼                                   │
│                    ┌──────────────────┐                         │
│                    │  Two-Stage Query │                         │
│                    │  Theme → Entity  │                         │
│                    └────────┬─────────┘                         │
│                             ▼                                    │
│                    ┌──────────────────┐                         │
│                    │   LLM Response   │                         │
│                    └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/haoohu/Cog-RAG.git
cd Cog-RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### From PyPI (Coming Soon)

```bash
pip install cograg
```

### Requirements

- Python >= 3.10
- See `requirements.txt` for full dependency list

## 🚀 Quick Start

### 1. Configure Your LLM API

Copy the configuration template and set your API credentials:

```bash
cp config_temp.py my_config.py
```

Edit `my_config.py`:

```python
# LLM Configuration
LLM_BASE_URL = "https://api.openai.com/v1"  # Or your custom endpoint
LLM_API_KEY = "your-api-key"
LLM_MODEL = "gpt-4o-mini"

# Embedding Configuration
EMB_BASE_URL = "https://api.openai.com/v1"
EMB_API_KEY = "your-api-key"
EMB_MODEL = "text-embedding-3-small"
EMB_DIM = 1536
```

### 2. Basic Usage

```python
import asyncio
import numpy as np
from cograg import CogRAG, QueryParam
from cograg.llm import openai_complete_if_cache, openai_embedding
from cograg.utils import EmbeddingFunc

# Configure LLM function
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="your-api-key",
        base_url="https://api.openai.com/v1",
        **kwargs,
    )

# Configure embedding function
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="text-embedding-3-small",
        api_key="your-api-key",
        base_url="https://api.openai.com/v1",
    )

# Initialize Cog-RAG
rag = CogRAG(
    working_dir="./my_rag_cache",
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=embedding_func
    ),
)

# Insert documents
with open("your_document.txt", "r") as f:
    text = f.read()
rag.insert(text)

# Query with different modes
# Full Cog-RAG (two-stage theme-entity alignment)
response = rag.query(
    "What are the main themes in this document?",
    param=QueryParam(mode="cog")
)
print(response)
```

### 3. Query Modes

Cog-RAG supports multiple query modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| `cog` | Full two-stage retrieval (Theme → Entity) | Best overall performance |
| `cog-hybrid` | Parallel theme and entity retrieval | Balanced approach |
| `cog-entity` | Entity-only retrieval | Detail-focused queries |
| `cog-theme` | Theme-only retrieval | High-level queries |
| `naive` | Traditional vector search | Baseline comparison |

```python
# Theme-focused query
response = rag.query(
    "What is the overall narrative?",
    param=QueryParam(mode="cog-theme")
)

# Entity-focused query  
response = rag.query(
    "What did Character X do?",
    param=QueryParam(mode="cog-entity")
)

# Hybrid query
response = rag.query(
    "How do the themes relate to the characters?",
    param=QueryParam(mode="cog-hybrid")
)
```

## 📖 Documentation

### Core Components

#### CogRAG Class

The main class for the Cog-RAG system:

```python
from cograg import CogRAG

rag = CogRAG(
    working_dir="./cache",           # Directory for storing indices
    chunk_token_size=1200,           # Size of text chunks
    chunk_overlap_token_size=100,    # Overlap between chunks
    llm_model_func=your_llm_func,    # Your LLM function
    embedding_func=your_embed_func,  # Your embedding function
    llm_model_max_async=16,          # Max concurrent LLM calls
)
```

#### QueryParam Class

Configure query behavior:

```python
from cograg import QueryParam

param = QueryParam(
    mode="cog",                      # Query mode
    only_need_context=False,         # Return only context (no LLM response)
    top_k=60,                        # Number of items to retrieve
    max_token_for_text_unit=1600,    # Max tokens for source text
    max_token_for_entity_context=300,# Max tokens for entity descriptions
    max_token_for_relation_context=1600,  # Max tokens for relationships
)
```

### Storage Architecture

Cog-RAG uses multiple storage backends:

- **JsonKVStorage**: Key-value storage for documents and chunks
- **NanoVectorDBStorage**: Vector database for similarity search
- **HypergraphStorage**: Hypergraph database for complex relationships

### Hypergraph Structure

#### Entity-Relation Hypergraph

- **Vertices**: Entities with type, description, and properties
- **Low-order Hyperedges**: Binary relationships between entity pairs
- **High-order Hyperedges**: Multi-entity relationships (N ≥ 3)

#### Key-Theme Hypergraph

- **Vertices**: Key entities with importance scores
- **Hyperedges**: Themes connecting multiple key entities

## 🔧 Advanced Configuration

### Custom LLM Integration

```python
from cograg.llm import openai_complete_if_cache

# Use Azure OpenAI
async def azure_llm_func(prompt, **kwargs):
    return await azure_openai_complete_if_cache(
        model="your-deployment",
        prompt=prompt,
        api_key="your-key",
        base_url="your-endpoint",
        **kwargs
    )

# Use local models (via API-compatible servers)
async def local_llm_func(prompt, **kwargs):
    return await openai_complete_if_cache(
        model="llama-3",
        prompt=prompt,
        base_url="http://localhost:8000/v1",
        **kwargs
    )
```

### Custom Embedding Models

```python
from cograg.utils import EmbeddingFunc

# Use sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

async def local_embedding(texts: list[str]):
    return model.encode(texts)

embedding_func = EmbeddingFunc(
    embedding_dim=384,
    max_token_size=512,
    func=local_embedding
)
```

## 📊 Benchmarks

Coming soon...

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and install dev dependencies
git clone https://github.com/haoohu/Cog-RAG.git
cd Cog-RAG
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black cograg/
isort cograg/

# Type checking
mypy cograg/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use Cog-RAG in your research, please cite:

```bibtex
@article{cograg2024,
  title={Cog-RAG: Cognitive-Inspired Dual-Hypergraph with Theme Alignment Retrieval-Augmented Generation},
  author={},
  journal={},
  year={2024}
}
```

## 🙏 Acknowledgements

- [nano-vectordb](https://github.com/gusye1234/nano-vectordb) for lightweight vector storage
- [hypergraph-db](https://github.com/iMoonLab/hypergraph-db) for hypergraph storage
- [OpenAI](https://openai.com/) for LLM and embedding APIs

## 📧 Contact

For questions or feedback, please open an issue or contact us at [your-email@example.com].

---

<div align="center">

**⭐ Star us on GitHub if you find this project useful! ⭐**

</div>

---

## 概述

**Cog-RAG** 是一个认知启发的检索增强生成框架，利用双超图结构和主题对齐机制来增强知识检索和问答能力。与依赖简单向量相似度搜索的传统 RAG 系统不同，Cog-RAG 通过超图建模复杂的多实体关系，并实现了受人类认知过程启发的两阶段主题-实体对齐机制。

### ✨ 主要特性

- 🔗 **双超图架构**：分离实体层和主题层的知识表示
- 🎯 **高阶关系建模**：捕获超越二元边的多实体关系
- 🧠 **认知启发的两阶段检索**：先主题对齐，后实体对齐
- 🔄 **多种查询模式**：支持不同使用场景的多种检索策略
- 📦 **易于集成**：简洁的 API 设计，兼容各种 LLM 提供商

详细的中文文档请参考 [docs/README_zh.md](docs/README_zh.md)。






