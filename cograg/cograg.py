# -*- coding: utf-8 -*-
"""CogRAG - Cognitive-Inspired Dual-Hypergraph RAG System.

Implements dual-hypergraph architecture combining:
- Entity-Relation Hypergraph: Complex multi-entity relationships
- Key-Theme Hypergraph: Thematic knowledge organization
"""

import os
import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, Union, List, Optional, cast

from .operate import (
    chunking_by_token_size,
    extract_entities,
    cog_theme_query,
    cog_entity_query,
    cog_query,
    cog_hybrid_query,
    naive_query,
)
from .llm import (
    gpt_4o_mini_complete,
    openai_embedding,
)
from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    HypergraphStorage,
)
from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    logger,
    set_logger,
)
from .base import (
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
    BaseHypergraphStorage,
)


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create an asyncio event loop."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        logger.info("Creating a new event loop in main thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


@dataclass
class CogRAG:
    """Dual-hypergraph RAG system with theme alignment.
    
    Query modes: "cog" (dual), "cog-hybrid", "cog-entity", "cog-theme", "naive".
    """
    
    # Working directory
    working_dir: str = field(
        default_factory=lambda: f"./CogRAG_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    # Logging configuration
    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # Text chunking parameters
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # Entity extraction parameters
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    entity_additional_properties_to_max_tokens: int = 250
    relation_summary_to_max_tokens: int = 750
    relation_keywords_to_max_tokens: int = 100

    # Embedding configuration
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM configuration
    llm_model_func: callable = gpt_4o_mini_complete
    llm_model_name: str = ""
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    vector_db_storage_cls_kt: Type[BaseVectorStorage] = NanoVectorDBStorage
    hypergraph_storage_cls: Type[BaseHypergraphStorage] = HypergraphStorage
    hypergraph_storage_cls_kt: Type[BaseHypergraphStorage] = HypergraphStorage
    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self) -> None:
        """Initialize storage backends and logging."""
        log_file = os.path.join(self.working_dir, "CogRAG.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"CogRAG init with param:\n  {_print_config}\n")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.chunk_entity_relation_hypergraph = self.hypergraph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.chunk_key_theme_hypergraph = self.hypergraph_storage_cls_kt(
            namespace="chunk_key_theme", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )

        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
        self.key_vdb = self.vector_db_storage_cls_kt(
            namespace="keys",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"key_name"},
        )
        self.relationships_vdb = self.vector_db_storage_cls(
            namespace="relationships",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"id_set"},
        )
        self.themes_vdb = self.vector_db_storage_cls_kt(
            namespace="themes",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"keys"},
        )
        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )

    def insert(self, string_or_strings: Union[str, List[str]]) -> None:
        """Insert documents into the RAG system."""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings: Union[str, List[str]]) -> None:
        """Asynchronously insert documents into the RAG system.
        
        Args:
            string_or_strings: Single document string or list of documents.
        """
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            # ----------------------------------------------------------------------------
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = {}
            for doc_key, doc in new_docs.items():
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            # ----------------------------------------------------------------------------
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            await self.chunks_vdb.upsert(inserting_chunks)
            # ----------------------------------------------------------------------------
            logger.info("[Entity Extraction]...")
            maybe_new_kg,key_theme = await extract_entities(
                inserting_chunks,
                knowledge_hypergraph_inst=self.chunk_entity_relation_hypergraph,
                knowledge_hypergraph_theme=self.chunk_key_theme_hypergraph,
                entity_vdb=self.entities_vdb,
                key_vdb = self.key_vdb,
                relationships_vdb=self.relationships_vdb,
                themes_vdb = self.themes_vdb,
                global_config=asdict(self),
            )
            if maybe_new_kg is None:
                logger.warning("No new entities and relationships found")
                return
            # ----------------------------------------------------------------------------
            self.chunk_entity_relation_hypergraph = maybe_new_kg
            self.chunk_key_theme_hypergraph = key_theme
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_done(self) -> None:
        """Trigger storage callbacks after indexing."""
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.key_vdb,
            self.relationships_vdb,
            self.themes_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_hypergraph,
            self.chunk_key_theme_hypergraph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def query(self, query: str, param: QueryParam = QueryParam()) -> str:
        """Query the RAG system for information."""
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()) -> str:
        """Asynchronously query the RAG system.
        
        Supported modes: "cog", "cog-hybrid", "cog-entity", "cog-theme", "naive".
        """
        if param.mode == "cog":
            response = await cog_query(
                query,
                self.chunk_entity_relation_hypergraph,
                self.chunk_key_theme_hypergraph,
                self.entities_vdb,
                self.key_vdb,
                self.relationships_vdb,
                self.themes_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )

        elif param.mode == "cog-hybrid":
            response = await cog_hybrid_query(
                query,
                self.chunk_entity_relation_hypergraph,
                self.chunk_key_theme_hypergraph,
                self.entities_vdb,
                self.key_vdb,
                self.relationships_vdb,
                self.themes_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )            
        elif param.mode == "cog-entity":
            response = await cog_entity_query(
                query,
                self.chunk_entity_relation_hypergraph,
                self.entities_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "cog-theme":
            response = await cog_theme_query(
                query,
                self.chunk_key_theme_hypergraph,
                self.key_vdb,
                self.themes_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
            )
        else:
            logger.error(f"Unknown query mode: {param.mode}")
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def _query_done(self) -> None:
        """Trigger storage callbacks after querying."""
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).query_done_callback())
        await asyncio.gather(*tasks)
