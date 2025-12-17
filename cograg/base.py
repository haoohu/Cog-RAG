# -*- coding: utf-8 -*-
"""
Base classes and type definitions for Cog-RAG.

This module defines the abstract base classes for storage backends and
the core data structures used throughout the Cog-RAG system.
"""

from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar, Any, Tuple, List, Set, Optional, Dict

from cograg.utils import EmbeddingFunc


class TextChunkSchema(TypedDict):
    """Schema for text chunk data structure."""
    tokens: int
    content: str
    full_doc_id: str
    chunk_order_index: int


T = TypeVar("T")


@dataclass
class QueryParam:
    """Query configuration parameters."""
    mode: Literal["cog", "cog-hybrid", "cog-entity", "cog-theme", "naive"] = "cog"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    top_k: int = 60
    max_token_for_text_unit: int = 4000
    max_token_for_entity_context: int = 4000
    max_token_for_relation_context: int = 4000


@dataclass
class StorageNameSpace:
    """Base class for namespaced storage with callback hooks."""
    namespace: str
    global_config: dict

    async def index_done_callback(self) -> None:
        """Hook called after indexing completes."""
        pass

    async def query_done_callback(self) -> None:
        """Hook called after querying completes."""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    """Abstract base class for vector storage with similarity search."""
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> List[Dict]:
        """Query for similar items."""
        raise NotImplementedError

    async def upsert(self, data: Dict[str, Dict]) -> None:
        """Insert or update vectors based on 'content' field."""
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    """Abstract base class for key-value storage backends."""

    async def all_keys(self) -> List[str]:
        """Get all keys in storage."""
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        """Get value by key."""
        raise NotImplementedError

    async def get_by_ids(
        self, ids: List[str], fields: Union[Set[str], None] = None
    ) -> List[Union[T, None]]:
        """Get multiple values by keys."""
        raise NotImplementedError

    async def filter_keys(self, data: List[str]) -> Set[str]:
        """Return keys that don't exist in storage."""
        raise NotImplementedError

    async def upsert(self, data: Dict[str, T]) -> None:
        """Insert or update key-value pairs."""
        raise NotImplementedError

    async def drop(self) -> None:
        """Clear all data from storage."""
        raise NotImplementedError


@dataclass
class BaseHypergraphStorage(StorageNameSpace):
    """Abstract base class for hypergraph storage (vertices and hyperedges)."""

    async def has_vertex(self, v_id: Any) -> bool:
        """Check if a vertex exists."""
        raise NotImplementedError

    async def has_hyperedge(self, e_tuple: Union[List, Set, Tuple]) -> bool:
        """Check if a hyperedge exists."""
        raise NotImplementedError

    async def get_vertex(self, v_id: str, default: Any = None) -> Optional[Dict]:
        """Get vertex data by ID."""
        raise NotImplementedError

    async def get_hyperedge(
        self, e_tuple: Union[List, Set, Tuple], default: Any = None
    ) -> Optional[Dict]:
        """Get hyperedge data."""
        raise NotImplementedError

    async def get_all_vertices(self) -> List:
        """Get all vertices in the hypergraph."""
        raise NotImplementedError

    async def get_all_hyperedges(self) -> List:
        """Get all hyperedges in the hypergraph."""
        raise NotImplementedError

    async def get_num_of_vertices(self) -> int:
        """Get the number of vertices."""
        raise NotImplementedError

    async def get_num_of_hyperedges(self) -> int:
        """Get the number of hyperedges."""
        raise NotImplementedError

    async def upsert_vertex(self, v_id: Any, v_data: Optional[Dict] = None) -> None:
        """Insert or update a vertex."""
        raise NotImplementedError

    async def upsert_hyperedge(
        self, e_tuple: Union[List, Set, Tuple], e_data: Optional[Dict] = None
    ) -> None:
        """Insert or update a hyperedge."""
        raise NotImplementedError

    async def remove_vertex(self, v_id: Any) -> None:
        """Remove a vertex from the hypergraph."""
        raise NotImplementedError

    async def remove_hyperedge(self, e_tuple: Union[List, Set, Tuple]) -> None:
        """Remove a hyperedge from the hypergraph."""
        raise NotImplementedError

    async def vertex_degree(self, v_id: Any) -> int:
        """Get the degree of a vertex (number of incident hyperedges)."""
        raise NotImplementedError

    async def hyperedge_degree(self, e_tuple: Union[List, Set, Tuple]) -> int:
        """Get the degree of a hyperedge (number of vertices)."""
        raise NotImplementedError

    async def get_nbr_e_of_vertex(self, v_id: Any) -> List:
        """Get all hyperedges incident to a vertex."""
        raise NotImplementedError

    async def get_nbr_v_of_hyperedge(
        self, e_tuple: Union[List, Set, Tuple]) -> List:
        """Get all vertices in a hyperedge."""
        raise NotImplementedError

    async def get_nbr_v_of_vertex(self, v_id: Any, exclude_self: bool = True) -> List:
        """Get neighboring vertices of a vertex."""
        raise NotImplementedError