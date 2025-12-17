# Cog-RAG Examples

This directory contains example scripts demonstrating how to use Cog-RAG.

## Demo Script: `cograg_demo.py`

A comprehensive example showing:
- Text chunking and entity extraction
- Building dual-hypergraph structures
- Running different query modes
- Generating responses with LLM

### Prerequisites

1. Set up your configuration by copying and editing the template:
   ```bash
   cp config_temp.py my_config.py
   # Edit my_config.py with your API keys and configuration
   ```

2. Install Cog-RAG:
   ```bash
   pip install -e ..
   ```

### Running the Demo

```bash
python cograg_demo.py
```

The demo will:
1. Load configuration from `my_config.py`
2. Process sample text data
3. Extract entities, relationships, and themes
4. Build the dual-hypergraph structure
5. Demonstrate query modes:
   - `cog`: Two-stage theme-entity retrieval
   - `cog-entity`: Entity-focused retrieval
   - `cog-theme`: Theme-focused retrieval
   - `cog-hybrid`: Combined entity and theme retrieval
   - `naive`: Simple vector similarity search

## Understanding the Architecture

### Input: Text Data (`mock_data.txt`)
- Contains sample documents to be processed
- Will be chunked into smaller segments
- Used for entity and theme extraction

### Processing Pipeline
1. **Chunking**: Text is split into manageable chunks by token size
2. **Extraction**: Entities, relationships, and themes are extracted using LLM
3. **Graph Building**: Dual-hypergraph structures are constructed
4. **Query**: Various retrieval strategies can be applied
5. **Response**: LLM generates responses based on retrieved context

### Output
- Entity-Relation Hypergraph: Models low-level and high-order relationships
- Key-Theme Hypergraph: Models theme-level knowledge
- Query Results: Retrieved context and LLM-generated responses

## Configuration

Edit `my_config.py` to customize:
- LLM provider and model
- Embedding model
- Query parameters (top_k, max tokens, etc.)
- Logging settings
- Cache locations

## Extending the Examples

To create your own example:

1. Copy `cograg_demo.py` and modify:
   ```python
   from cograg import CogRAG, QueryParam
   
   # Load configuration
   from my_config import LLM_BASE_URL, LLM_API_KEY, EMB_BASE_URL, EMB_API_KEY
   
   # Create CogRAG instance
   cograg = CogRAG(
       llm_model_func=your_llm_func,
       embedding_func=your_embedding_func,
       global_config=your_config
   )
   
   # Process your data
   # Create queries
   # Get responses
   ```

2. Customize the LLM and embedding functions for your use case

3. Prepare your own text data

4. Run your example script

## Troubleshooting

- **API Key Issues**: Make sure `my_config.py` has valid API keys
- **Import Errors**: Ensure Cog-RAG is installed in development mode
- **Memory Issues**: Reduce `max_token_size` or `top_k` for smaller datasets
- **LLM Timeouts**: Increase timeout settings in configuration

## Tips for Production Use

1. Use proper error handling and logging
2. Implement caching for expensive operations
3. Test with your actual data before deployment
4. Monitor LLM API usage and costs
5. Use appropriate batch sizes for large datasets
6. Consider implementing retry logic with exponential backoff

## Questions or Issues?

If you encounter problems or have suggestions, please open an issue on the [GitHub repository](https://github.com/haoohu/Cog-RAG/issues).
