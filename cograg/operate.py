"""Core operations: text chunking, entity/theme extraction, hypergraph construction,
and query modes (cog, cog-hybrid, cog-entity, cog-theme, naive)."""

import sys
import asyncio
import json
import re
import logging
from datetime import datetime
from typing import Union, List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import warnings
import time

from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
)
from .base import (
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
    BaseHypergraphStorage,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


# --- Text Chunking ---

def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


# --- Summary Functions ---

async def _handle_entity_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:
        return description
        
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    
    if summary is None:
        logger.warning("Entity description summary not found")
        summary = use_description
    return summary

# summarize the additional properties of the entity
async def _handle_entity_additional_properties(
    entity_name: str,
    additional_properties: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_additional_properties_to_max_tokens"]

    tokens = encode_string_by_tiktoken(additional_properties, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:
        return additional_properties
        
    prompt_template = PROMPTS["summarize_entity_additional_properties"]
    use_additional_properties = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_name,
        additional_properties_list=use_additional_properties.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    
    if summary is None:
        logger.warning("Entity additional_properties summary not found")
        summary = use_additional_properties
    return summary

# summarize the descriptions of the relation
async def _handle_relation_summary(
    relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["relation_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:
        return description
        
    prompt_template = PROMPTS["summarize_relation_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        relation_name=relation_name,
        relation_description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    
    if summary is None:
        logger.warning("Relation description summary not found")
        summary = use_description
    return summary

# summarize the keywords of the relation
async def _handle_relation_keywords_summary(
    relation_name: str,
    keywords: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["relation_keywords_to_max_tokens"]

    tokens = encode_string_by_tiktoken(keywords, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:
        return keywords
        
    prompt_template = PROMPTS["summarize_relation_keywords"]
    use_keywords = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        relation_name=relation_name,
        keywords_list=use_keywords.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    
    if summary is None:
        logger.warning("Relation keywords summary not found")
        summary = use_keywords
    return summary

async def _handle_single_entity_extraction(
    record_attributes: List[str],
    chunk_key: str,
) -> Optional[Dict[str, Any]]:
    if len(record_attributes) < 4 or record_attributes[0] != '"Entity"':
        return None
        
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
        
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    entity_additional_properties = clean_str(record_attributes[4:])

    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
        additional_properties=entity_additional_properties,
    )


async def _handle_single_key_extraction(
    record_attributes: List[str],
    chunk_key: str,
) -> Optional[Dict[str, Any]]:
    if len(record_attributes) < 4 or record_attributes[0] != '"key_entity"':
        return None
        
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
        
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 75.0
    )

    return dict(
        key_name=entity_name,
        key_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
        weight=weight,
    )


async def _merge_keys_then_upsert(
    entity_name: str,
    nodes_data: List[Dict],
    knowledge_hypergraph_theme,
    global_config: dict,
) -> Dict[str, Any]:
    """Merge key concept data and upsert to hypergraph."""
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_hypergraph_theme.get_vertex(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["key_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["key_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    
    # Combine descriptions
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    
    # Summarize if needed
    description = await _handle_entity_summary(
        entity_name, description, global_config
    )

    node_data = dict(
        key_type=entity_type,
        description=description,
        source_id=source_id,
    )
    
    await knowledge_hypergraph_theme.upsert_vertex(entity_name, node_data)
    node_data["key_name"] = entity_name
    return node_data


async def _handle_single_theme_extraction(
    record_attributes: List[str],
    chunk_key: str,
) -> Optional[Dict[str, Any]]:
    """Extract a single theme from record attributes."""
    if len(record_attributes) > 4 or record_attributes[0] != '"theme"':
        return None
        
    theme_text = clean_str(record_attributes[1].upper())
    if not theme_text.strip():
        return None
        
    entity_source_id = chunk_key
    return dict(
        theme_text=theme_text,
        source_id=entity_source_id,
    )

async def _handle_single_relationship_extraction_low(
    record_attributes: List[str],
    chunk_key: str,
) -> Optional[Dict[str, Any]]:
    """Extract a low-order hyperedge from record attributes."""
    if len(record_attributes) < 6 or record_attributes[0] != '"Low-order Hyperedge"':
        return None
    # add this record as hyperedge
    entity_num = len(record_attributes) - 3
    entities = []
    for i in range(1, entity_num):
        entities.append(clean_str(record_attributes[i].upper()))
    edge_description = clean_str(record_attributes[-3])

    edge_keywords = clean_str(record_attributes[-2])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 0.75
    )
    return dict(
        entityN=entities,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
        level_hg="Low-order Hyperedge",
    )

async def _handle_single_relationship_extraction_high(
    record_attributes: List[str],
    chunk_key: str,
):
    if len(record_attributes) < 7 or record_attributes[0] != '"High-order Hyperedge"':
        return None
    # add this record as hyperedge
    entity_num = len(record_attributes) - 4
    entities = []
    for i in range(1, entity_num):
        entities.append(clean_str(record_attributes[i].upper()))
    edge_description = clean_str(record_attributes[-4])
    edge_keywords = clean_str(record_attributes[-2])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 0.75
    )
    return dict(
        entityN=entities,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
        level_hg="High-order Hyperedge",
    )





async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: List[Dict],
    knowledge_hypergraph_inst: BaseHypergraphStorage,
    global_config: dict,
) -> Dict[str, Any]:
    """Merge entity node data and upsert to entity-relation hypergraph.
    
    Args:
        entity_name: Name of the entity vertex.
        nodes_data: List of node data dictionaries to merge.
        knowledge_hypergraph_inst: Entity-relation hypergraph storage.
        global_config: Global configuration dictionary.
    
    Returns:
        Merged node data dictionary with entity_name included.
    """
    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_additional_properties = []

    already_node = await knowledge_hypergraph_inst.get_vertex(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])
        already_additional_properties.append(already_node["additional_properties"])


    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    # nodes_data = [dp["description"] for dp in nodes_data if dp["description"] is not None]
    description = GRAPH_FIELD_SEP.join(

        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    additional_properties = GRAPH_FIELD_SEP.join(
        sorted(set(
            prop
            for dp in nodes_data
            for prop in dp["additional_properties"]
        ) | set(already_additional_properties))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_summary(
        entity_name, description, global_config
    )
    additional_properties = await _handle_entity_additional_properties(
        entity_name, additional_properties, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        additional_properties=additional_properties,
    )
    await knowledge_hypergraph_inst.upsert_vertex(
        entity_name,
        node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    id_set: Tuple,
    edges_data: List[Dict],
    knowledge_hypergraph_inst: BaseHypergraphStorage,
    global_config: dict,
) -> Dict[str, Any]:
    """Merge hyperedge data and upsert to entity-relation hypergraph.
    
    Args:
        id_set: Tuple of entity names forming the hyperedge.
        edges_data: List of edge data dictionaries to merge.
        knowledge_hypergraph_inst: Entity-relation hypergraph storage.
        global_config: Global configuration dictionary.
    
    Returns:
        Merged edge data dictionary with id_set included.
    """
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_hypergraph_inst.has_hyperedge(id_set):
        already_edge = await knowledge_hypergraph_inst.get_hyperedge(id_set)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )

    for need_insert_id in id_set:
        if not (await knowledge_hypergraph_inst.has_vertex(need_insert_id)):
            await knowledge_hypergraph_inst.upsert_vertex(
                need_insert_id,
                {
                    "source_id": source_id,
                    "description": "UNKNOWN",
                    "additional_properties": "UNKNOWN",
                    "entity_type": "UNKNOWN",
                },
            )
    description = await _handle_relation_summary(
        id_set, description, global_config
    )
    filter_keywords = await _handle_relation_keywords_summary(
        id_set, keywords, global_config
    )

    await knowledge_hypergraph_inst.upsert_hyperedge(
        id_set,
        dict(
            description=description,
            keywords=filter_keywords,
            source_id=source_id,
            weight=weight
        ),
    )

    edge_data = dict(
        id_set=id_set,
        description=description,
        keywords=filter_keywords,
    )

    return edge_data


async def _merge_themes_then_upsert(
    id_set: Tuple,
    edges_data: List[Dict],
    knowledge_hypergraph_theme: BaseHypergraphStorage,
    global_config: dict,
) -> Dict[str, Any]:
    """Merge theme hyperedge data and upsert to key-theme hypergraph.
    
    Args:
        id_set: Tuple of key names forming the theme hyperedge.
        edges_data: List of theme data dictionaries to merge.
        knowledge_hypergraph_theme: Key-theme hypergraph storage.
        global_config: Global configuration dictionary.
    
    Returns:
        Merged theme data dictionary with keys included.
    """
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_hypergraph_theme.has_hyperedge(id_set):
        already_edge = await knowledge_hypergraph_theme.get_hyperedge(id_set)
        # already_weights.append(already_edge["weight"])
        # already_source_ids.extend(
        #     split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        # )
        already_description.append(already_edge["theme_text"])
        # already_keywords.extend(
        #     split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        # )

    # weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["theme_text"] for dp in edges_data] + already_description))
    )
    # keywords = GRAPH_FIELD_SEP.join(
    #     sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    # )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data])
    )

    await knowledge_hypergraph_theme.upsert_hyperedge(
        id_set,
        dict(
            theme_text=description,
            source_id=source_id,
        ),
    )


    edge_data = dict(
        keys=id_set,
        theme_text=description,
        source_id=source_id,
    )

    return edge_data


async def extract_entities(
    chunks: Dict[str, TextChunkSchema],
    knowledge_hypergraph_inst: BaseHypergraphStorage,
    knowledge_hypergraph_theme: BaseHypergraphStorage,
    entity_vdb: BaseVectorStorage,
    key_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    themes_vdb: BaseVectorStorage,
    global_config: dict,
) -> Optional[Tuple[BaseHypergraphStorage, BaseHypergraphStorage]]:
    """Extract entities and themes from text chunks, build dual-hypergraph.
    
    This function processes text chunks to extract:
    - Entities and relationships for the entity-relation hypergraph
    - Keys and themes for the key-theme hypergraph
    
    Args:
        chunks: Dictionary of text chunks to process.
        knowledge_hypergraph_inst: Entity-relation hypergraph storage.
        knowledge_hypergraph_theme: Key-theme hypergraph storage.
        entity_vdb: Vector database for entity embeddings.
        key_vdb: Vector database for key concept embeddings.
        relationships_vdb: Vector database for relationship embeddings.
        themes_vdb: Vector database for theme embeddings.
        global_config: Global configuration dictionary.
    
    Returns:
        Tuple of (entity_hypergraph, theme_hypergraph) or None if extraction fails.
    """
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entity_extraction"]
    theme_extract_prompt = PROMPTS["theme_extraction"]
    # We can choose the example what we want from the prompt.
    example_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
    )
    example_prompt = PROMPTS["entity_extraction_examples"][3]
    theme_example_prompt = PROMPTS["theme_extraction_examples"][0]
    example_str = example_prompt.format(**example_base)
    theme_str = theme_example_prompt.format(**example_base)

    context_base = dict(
        language=PROMPTS["DEFAULT_LANGUAGE"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples = example_str
    )
    theme_context_base = dict(
        language=PROMPTS["DEFAULT_LANGUAGE"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        examples = theme_str
    )

    already_processed = 0
    already_entities = 0
    already_relations = 0
    already_relations_low = 0
    already_relations_high = 0
    already_theme = 0
    already_key = 0
    
    async def _process_single_content(chunk_key_dp: Tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations, already_relations_low, already_relations_high, already_theme, already_key

        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        theme_prompt = theme_extract_prompt.format(**theme_context_base, input_text=content)

        final_result = await use_llm_func(hint_prompt)
        if final_result is None:
            return None, None, None, None, None, None
        theme_result = await use_llm_func(theme_prompt)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        theme_records = split_string_by_multi_markers(
            theme_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        maybe_edges_low = defaultdict(list)
        maybe_edges_high = defaultdict(list)
        maybe_theme = defaultdict(list)
        maybe_key = defaultdict(list)
        maybe_theme_key = []
        if_theme = None
        for theme_record in theme_records:
            if len(theme_record)<10:
                continue
            if theme_record[0]=='(' and theme_record[-1]!=')':
                theme_record=theme_record+')'
            if theme_record[0]!='(' and theme_record[-1]==')':
                theme_record='('+theme_record
            if theme_record[0]=='{' and theme_record[-1]=='}':
                theme_record='('+theme_record[1:-1]+')'
            theme_record1 = re.search(r"\((.*)\)", theme_record)

            if theme_record1 is not None:
                theme_record = theme_record1.group(1)
            # theme_record = theme_record.group(1)
            theme_record_attributes = split_string_by_multi_markers(
                theme_record, [context_base["tuple_delimiter"]]
            )

            if_key = await _handle_single_key_extraction(
                theme_record_attributes, chunk_key
            )

            if if_key is not None:
                maybe_key[if_key["key_name"]].append(if_key)
                maybe_theme_key.append(if_key["key_name"])
                continue
            if if_theme is None:
                if_theme = await _handle_single_theme_extraction(
                    theme_record_attributes, chunk_key
                )
            else:
                if_theme_re = await _handle_single_theme_extraction(
                    theme_record_attributes, chunk_key
                )
                if if_theme_re is not None:
                    if if_theme['theme_text'] is None:
                        if_theme['theme_text'] = if_theme_re['theme_text']
                    if isinstance(if_theme['theme_text'], str):
                        if_theme['theme_text'] = [if_theme['theme_text']].append(if_theme_re['theme_text'])
                    if isinstance(if_theme['theme_text'], list):
                        if_theme['theme_text'] = if_theme['theme_text'].append(if_theme_re['theme_text'])
                # raise Exception('theme error!')
                # maybe_theme[if_theme["source_id"]].append(if_theme)
        if if_theme is None:
            if_theme = dict(
                theme_text=clean_str(theme_records[0].upper()),
                source_id=chunk_key,
            )
        if_theme["keys"] = tuple(maybe_theme_key)
        if if_theme['theme_text'] is not None:
            maybe_theme[if_theme["keys"]].append(if_theme)
        for record in records:
            
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction_low(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[tuple((if_relation["entityN"]))].append(
                    if_relation
                )
                maybe_edges_low[tuple((if_relation["entityN"]))].append(
                    if_relation
                )

            if_relation = await _handle_single_relationship_extraction_high(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[tuple((if_relation["entityN"]))].append(
                    if_relation
                )
                maybe_edges_high[tuple((if_relation["entityN"]))].append(
                    if_relation
                )
        
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        already_relations_low += len(maybe_edges_low)
        already_relations_high += len(maybe_edges_high)
        already_theme += len(maybe_theme)
        already_key += len(maybe_key)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]

        # 计算用时
        current_time = datetime.now()
        time = current_time - begin_time
        total_seconds = int(time.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        # 进度条
        percent = (already_processed / len(ordered_chunks)) * 100
        bar_length = int(50 * already_processed // len(ordered_chunks))
        bar = '█' * bar_length + '-' * (50 - bar_length)
        sys.stdout.write(
            f'\n\r|{bar}| {percent:.2f}% |{hours:02}:{minutes:02}:{seconds:02}| {now_ticks} Processed {already_theme} themes, {already_key} keys, {already_entities} entities, {already_relations} relations, {already_relations_low} relations_low, {already_relations_high} relations_high \n')
        sys.stdout.flush()
        return dict(maybe_nodes), dict(maybe_edges), dict(maybe_edges_low), dict(maybe_edges_high), dict(maybe_theme) , dict(maybe_key)

    # ----------------------------------------------------------------------------
    # use_llm_func is wrapped in asyncio.Semaphore for concurrency control
    begin_time = datetime.now()

    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks ]
    )
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    high = defaultdict(list)
    low = defaultdict(list)
    maybe_themes = defaultdict(list)
    maybe_keys = defaultdict(list)
    for m_nodes, m_edges, low_edge, high_edge,m_theme,m_key in results:
        if m_nodes is not None:
            for k, v in m_nodes.items():
                maybe_nodes[k].extend(v)
        if m_edges is not None:
            for k, v in m_edges.items():
                maybe_edges[tuple(sorted(k))].extend(v)
        if low_edge is not None:
            for k, v in low_edge.items():
                low[tuple(sorted(k))].extend(v)
        if high_edge is not None:
            for k, v in high_edge.items():
                high[tuple(sorted(k))].extend(v)
        if m_theme is not None:
            for k, v in m_theme.items():
                maybe_themes[tuple(sorted(k))].extend(v)
        if m_key is not None:
            for k, v in m_key.items():
                maybe_keys[k].extend(v)
        if m_nodes is None or m_edges is None or low_edge is None or high_edge is None or m_theme is None or m_key is None:
            logger.warning("Extraction returned None for one or more elements")
    # ----------------------------------------------------------------------------
    """
        update the hypergraph database
    """
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_hypergraph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )

    all_keys_data = await asyncio.gather(
        *[
            _merge_keys_then_upsert(k, v, knowledge_hypergraph_theme, global_config)
            for k, v in maybe_keys.items()
        ]
    )

    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k, v, knowledge_hypergraph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    all_theme_data = await asyncio.gather(
        *[
            _merge_themes_then_upsert(k, v, knowledge_hypergraph_theme, global_config)
            for k, v in maybe_themes.items()
        ]
    )

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if not len(all_relationships_data):
        logger.warning(
            "Didn't extract any relationships, maybe your LLM is not working"
        )
    if not len(all_keys_data):
        logger.warning("Didn't extract any keys, maybe your LLM is not working")
        return None
    if not len(all_theme_data):
        logger.warning(
            "Didn't extract any themes, maybe your LLM is not working"
        )
        return None

    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if key_vdb is not None:
        key_for_vdb = {
            compute_mdhash_id(dp["key_name"], prefix="ent-"): {
                "content": dp["key_name"] + dp["description"],
                "key_name": dp["key_name"],
            }
            for dp in all_keys_data
        }
        await key_vdb.upsert(key_for_vdb)
    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(str(sorted(dp["id_set"])), prefix="rel-"): {
                "id_set": dp["id_set"],
                "content": dp["keywords"]
                           + str(dp["id_set"])
                           + dp["description"],
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)
    if themes_vdb is not None:
        theme_for_vdb = {
            compute_mdhash_id(str(sorted(dp["keys"])), prefix="rel-"): {
                "keys": dp["keys"],
                "content": dp["source_id"]
                           + str(dp["keys"])
                           + dp["theme_text"],
            }
            for dp in all_theme_data
        }
        await themes_vdb.upsert(theme_for_vdb)

    return knowledge_hypergraph_inst, knowledge_hypergraph_theme


async def _build_entity_query_context(
    query: str,
    knowledge_hypergraph_inst: BaseHypergraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
) -> Optional[str]:
    """Build query context from entity-relation hypergraph.
    
    Args:
        query: Query string for entity retrieval.
        knowledge_hypergraph_inst: Entity-relation hypergraph storage.
        entities_vdb: Vector database for entity embeddings.
        text_chunks_db: Key-value storage for text chunks.
        query_param: Query parameters.
    
    Returns:
        Formatted context string with entities, relationships, and sources.
    """
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_hypergraph_inst.get_vertex(r["entity_name"]) for r in results]
    )

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_hypergraph_inst.vertex_degree(r["entity_name"]) for r in results]
    )

    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]

    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_hypergraph_inst
    )

    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_hypergraph_inst
    )

    logger.info(
        f"entity query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entities_section_list = [["id", "entity", "type", "description", "additional properties", "rank"]]
    for i, n in enumerate(node_datas):
        entities_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n.get("additional_properties", "UNKNOWN"),
                n["rank"],
            ]
        )

    entities_context = list_of_list_to_csv(entities_section_list)

    relations_section_list = [
        ["id", "entity set", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )

    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _find_most_related_text_unit_from_entities(
    node_datas: List[Dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_hypergraph_inst: BaseHypergraphStorage,
) -> List[TextChunkSchema]:
    """Find most related text units from entity nodes.
    
    Args:
        node_datas: List of entity node data.
        query_param: Query parameters.
        text_chunks_db: Key-value storage for text chunks.
        knowledge_hypergraph_inst: Entity-relation hypergraph storage.
    
    Returns:
        List of text chunk data.
    """
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]

    edges = await asyncio.gather(
        *[knowledge_hypergraph_inst.get_nbr_e_of_vertex(dp['entity_name']) for dp in node_datas]
    )

    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        # this_edges 是超边元组的集合，每个超边是顶点元组如 ('v1', 'v2', ...)
        # 需要遍历每个超边中的顶点
        for edge_tuple in this_edges:
            all_one_hop_nodes.update(edge_tuple)

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_hypergraph_inst.get_vertex(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            if this_edges:  # Add check for None edges
                # this_edges 是超边元组的集合，遍历每个超边
                for edge_tuple in this_edges:
                    # edge_tuple 是顶点元组如 ('v1', 'v2', ...)，遍历其中的顶点
                    for e in edge_tuple:
                        if (
                            e in all_one_hop_text_units_lookup
                            and c_id in all_one_hop_text_units_lookup[e]
                        ):
                            relation_counts += 1

            chunk_data = await text_chunks_db.get_by_id(c_id)
            if chunk_data is not None and "content" in chunk_data:  # Add content check
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                    "relation_counts": relation_counts,
                }

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units,
        key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: List[Dict],
    query_param: QueryParam,
    knowledge_hypergraph_inst: BaseHypergraphStorage,
) -> List[Dict]:
    """Find most related hyperedges from entity nodes.
    
    Args:
        node_datas: List of entity node data.
        query_param: Query parameters.
        knowledge_hypergraph_inst: Entity-relation hypergraph storage.
    
    Returns:
        List of hyperedge data dictionaries.
    """
    all_related_edges = await asyncio.gather(
        *[knowledge_hypergraph_inst.get_nbr_e_of_vertex(dp['entity_name']) for dp in node_datas]
    )

    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_hypergraph_inst.get_hyperedge(e) for e in all_edges]
    )

    all_edges_degree = await asyncio.gather(
        *[knowledge_hypergraph_inst.hyperedge_degree(e) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v !=[]
    ]

    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_relation_context,
    )
    return all_edges_data


async def _build_relation_query_context(
    keywords: str,
    knowledge_hypergraph_theme: BaseHypergraphStorage,
    keys_vdb: BaseVectorStorage,
    themes_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
) -> Optional[str]:
    """Build query context from key-theme hypergraph.
    
    Args:
        keywords: Keywords string for theme retrieval.
        knowledge_hypergraph_theme: Key-theme hypergraph storage.
        keys_vdb: Vector database for key concept embeddings.
        themes_vdb: Vector database for theme embeddings.
        text_chunks_db: Key-value storage for text chunks.
        query_param: Query parameters.
    
    Returns:
        Formatted context string with entities, relationships, and sources.
    """
    results = await themes_vdb.query(keywords, top_k=query_param.top_k)
    if not len(results):
        return None

    edge_datas = await asyncio.gather(
        *[knowledge_hypergraph_theme.get_hyperedge(r['keys']) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(
        *[knowledge_hypergraph_theme.hyperedge_degree(e['keys']) for e in results]
    )

    edge_datas = [
        {"keys": k["keys"], "rank": d, **v}
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["theme_text"],
        max_token_size=query_param.max_token_for_relation_context,
    )

    use_entities = await _find_most_related_keys_from_themes(
        edge_datas, query_param, knowledge_hypergraph_theme
    )
    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_hypergraph_theme
    )
    logger.info(
        f"relation query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )
    relations_section_list = [
        ["id", "entity set", "description", "source_id",  "rank"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["keys"],
                e["theme_text"],
                e["source_id"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description",  "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["key_name"],
                n.get("key_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _find_most_related_keys_from_themes(
    edge_datas: List[Dict],
    query_param: QueryParam,
    knowledge_hypergraph_theme: BaseHypergraphStorage,
) -> List[Dict]:
    """Find most related key concepts from theme hyperedges.
    
    Args:
        edge_datas: List of theme hyperedge data.
        query_param: Query parameters.
        knowledge_hypergraph_theme: Key-theme hypergraph storage.
    
    Returns:
        List of key node data dictionaries.
    """
    entity_names = set()
    for e in edge_datas:
        for f in e["keys"]:
            if await knowledge_hypergraph_theme.has_vertex(f):
                entity_names.add(f)

    node_datas = await asyncio.gather(
        *[knowledge_hypergraph_theme.get_vertex(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_hypergraph_theme.vertex_degree(entity_name) for entity_name in entity_names]
    )

    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_entity_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: List[Dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_hypergraph_theme: BaseHypergraphStorage,
) -> List[TextChunkSchema]:
    """Find related text units from theme hyperedges.
    
    Args:
        edge_datas: List of theme hyperedge data.
        query_param: Query parameters.
        text_chunks_db: Key-value storage for text chunks.
        knowledge_hypergraph_theme: Key-theme hypergraph storage.
    
    Returns:
        List of text chunk data.
    """
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]

    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                }

    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]

    return all_text_units

# =============================================================================
# Query Functions
# =============================================================================

# Global variables for performance timing
TIME_LIST: List[float] = []
Time_begin: List[float] = []


async def cog_query(
    query: str,
    knowledge_hypergraph_inst: BaseHypergraphStorage,
    knowledge_hypergraph_theme: BaseHypergraphStorage,
    entities_vdb: BaseVectorStorage,
    keys_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    themes_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """Execute a two-stage cognition-guided query with theme alignment.
    
    This is the primary query mode using dual-hypergraph architecture:
    - Stage 1 (Theme-based): Extract theme keywords and retrieve theme context
    - Stage 2 (Entity-based): Extract entity keywords aligned with theme and 
                               retrieve entity context
    
    Query pipeline:
    1. Extract theme-keywords from initial query
    2. Build relation context from theme hypergraph
    3. Generate theme-aware response
    4. Extract entity-keywords aligned with theme response
    5. Build entity context from entity-relation hypergraph
    6. Generate final response with both context

    Returns:
        Generated response string or fail response if no context found.
    """
    use_model_func = global_config["llm_model_func"]
    
    # === Stage 1: Theme-based retrieval ===
    # Extract theme keywords from query
    try:
        kw_result = await use_model_func(
            PROMPTS["theme_keywords_extraction"].format(query=query)
        )
        kw_data = json.loads(kw_result)
    except json.JSONDecodeError:
        try:
            kw_result = "{" + kw_result.split("{")[1].split("}")[0] + "}"
            kw_data = json.loads(kw_result)
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"Failed to extract theme keywords: {e}")
            return PROMPTS["fail_response"]
    
    theme_keywords = ", ".join(kw_data.get("theme_keywords", []))
    if not theme_keywords:
        logger.warning("No theme keywords extracted for theme-based retrieval")
        return PROMPTS["fail_response"]
    
    # Build theme context from theme hypergraph
    theme_context = await _build_relation_query_context(
        theme_keywords,
        knowledge_hypergraph_theme,
        keys_vdb,
        themes_vdb,
        text_chunks_db,
        query_param,
    )
    
    if query_param.only_need_context:
        return theme_context
    if theme_context is None:
        logger.warning("No theme context retrieved")
        return PROMPTS["fail_response"]
    
    # Generate theme-aware response
    theme_sys_prompt = PROMPTS["rag_response"].format(
        context_data=theme_context, response_type=query_param.response_type
    )
    theme_define_str = PROMPTS["rag_define"].format(
        entity_keywords="", theme_keywords=theme_keywords
    )
    theme_response = await use_model_func(
        query + theme_define_str, system_prompt=theme_sys_prompt
    )
    # Clean response artifacts
    if len(theme_response) > len(theme_sys_prompt):
        theme_response = (
            theme_response.replace(theme_sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    
    # === Stage 2: Entity-based retrieval aligned with theme ===
    # Extract entity keywords aligned with theme response
    try:
        kw_result = await use_model_func(
            PROMPTS["entity_keywords_aglin"].format(query=query, theme=theme_response)
        )
        kw_data = json.loads(kw_result)
    except json.JSONDecodeError:
        try:
            kw_result = "{" + kw_result.split("{")[1].split("}")[0] + "}"
            kw_data = json.loads(kw_result)
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"Failed to extract aligned entity keywords: {e}")
            return PROMPTS["fail_response"]
    
    aligned_entity_kw = ", ".join(kw_data.get("entity_keywords", []))
    if not aligned_entity_kw:
        logger.warning("No entity keywords extracted in stage 2")
        return PROMPTS["fail_response"]
    
    # Retrieve entity context from entity hypergraph
    entity_context = await _build_entity_query_context(
        aligned_entity_kw,
        knowledge_hypergraph_inst,
        entities_vdb,
        text_chunks_db,
        query_param,
    )
    
    if query_param.only_need_context:
        return entity_context
    if entity_context is None:
        logger.warning("No entity context retrieved")
        return PROMPTS["fail_response"]
    
    # Generate final response with entity context and theme alignment
    final_sys_prompt = PROMPTS["rag_response"].format(
        context_data=entity_context, response_type=query_param.response_type
    )
    final_define_str = PROMPTS["rag_define_aglin"].format(
        query=query, theme_answer=theme_response, entity_keywords=aligned_entity_kw
    )
    final_response = await use_model_func(
        query + final_define_str, system_prompt=final_sys_prompt
    )
    # Clean response artifacts
    if len(final_response) > len(final_sys_prompt):
        final_response = (
            final_response.replace(final_sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return final_response


async def cog_entity_query(
    query: str,
    knowledge_hypergraph_inst: BaseHypergraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """Execute an entity-focused query using the entity-relation hypergraph.
    
    This query mode focuses on entity-level extraction and retrieval,
    ignoring theme-level context.
        
    Returns:
        Generated response string or fail response if no context found.
    """
    use_model_func = global_config["llm_model_func"]

    # Extract entity keywords from query
    try:
        kw_result = await use_model_func(PROMPTS["entity_keywords_extraction"].format(query=query))
        kw_data = json.loads(kw_result)
    except json.JSONDecodeError:
        try:
            kw_result = "{" + kw_result.split("{")[1].split("}")[0] + "}"
            kw_data = json.loads(kw_result)
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"Failed to extract keywords: {e}")
            return PROMPTS["fail_response"]
    
    entity_keywords = ", ".join(kw_data.get("entity_keywords", []))
    
    if not entity_keywords:
        logger.warning("No entity keywords extracted")
        return PROMPTS["fail_response"]
    
    # Retrieve entity context
    entity_context = await _build_entity_query_context(
        entity_keywords,
        knowledge_hypergraph_inst,
        entities_vdb,
        text_chunks_db,
        query_param,
    )
    
    if query_param.only_need_context:
        return entity_context
    if entity_context is None:
        logger.warning("No entity context retrieved")
        return PROMPTS["fail_response"]
    
    # Generate response
    sys_prompt = PROMPTS["rag_response"].format(
        context_data=entity_context, response_type=query_param.response_type
    )
    define_str = PROMPTS["rag_define"].format(
        entity_keywords=entity_keywords, theme_keywords=""
    )
    response = await use_model_func(query + define_str, system_prompt=sys_prompt)
    
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return response



async def cog_theme_query(
    query: str,
    knowledge_hypergraph_theme: BaseHypergraphStorage,
    keys_vdb: BaseVectorStorage,
    themes_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """Execute a theme-focused query using the theme hypergraph.
    
    This query mode focuses on theme-level retrieval,
    ignoring entity-level context.
        
    Returns:
        Generated response string or fail response if no context found.
    """
    use_model_func = global_config["llm_model_func"]

    # Extract theme keywords from query
    try:
        kw_result = await use_model_func(PROMPTS["theme_keywords_extraction"].format(query=query))
        kw_data = json.loads(kw_result)
    except json.JSONDecodeError:
        try:
            kw_result = "{" + kw_result.split("{")[1].split("}")[0] + "}"
            kw_data = json.loads(kw_result)
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"Failed to extract keywords: {e}")
            return PROMPTS["fail_response"]
    
    theme_keywords = ", ".join(kw_data.get("theme_keywords", []))
    
    if not theme_keywords:
        logger.warning("No theme keywords extracted")
        return PROMPTS["fail_response"]
    
    # Build theme context
    theme_context = await _build_relation_query_context(
        theme_keywords,
        knowledge_hypergraph_theme,
        keys_vdb,
        themes_vdb,
        text_chunks_db,
        query_param,
    )
    
    if query_param.only_need_context:
        return theme_context
    if theme_context is None:
        logger.warning("No theme context retrieved")
        return PROMPTS["fail_response"]
    
    # Generate response
    sys_prompt = PROMPTS["rag_response"].format(
        context_data=theme_context, response_type=query_param.response_type
    )
    define_str = PROMPTS["rag_define"].format(
        entity_keywords="", theme_keywords=theme_keywords
    )
    response = await use_model_func(query + define_str, system_prompt=sys_prompt)
    
    # Clean response artifacts
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return response


async def cog_hybrid_query(
    query: str,
    knowledge_hypergraph_inst: BaseHypergraphStorage,
    knowledge_hypergraph_theme: BaseHypergraphStorage,
    entities_vdb: BaseVectorStorage,
    keys_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    themes_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """Execute a hybrid query combining entity and theme retrieval.
    
    This query mode retrieves context from both the entity-relation
    hypergraph and the key-theme hypergraph, then combines them.
    
    Returns:
        Generated response string or fail response if no context found.
    """
    use_model_func = global_config["llm_model_func"]
    
    # Extract entity and theme keywords in parallel
    entity_kw_task = use_model_func(PROMPTS["entity_keywords_extraction"].format(query=query))
    theme_kw_task = use_model_func(PROMPTS["theme_keywords_extraction"].format(query=query))
    
    entity_kw_result, theme_kw_result = await asyncio.gather(entity_kw_task, theme_kw_task)
    
    # Parse entity keywords
    entity_keywords = ""
    try:
        entity_kw_data = json.loads(entity_kw_result)
        entity_keywords = ", ".join(entity_kw_data.get("entity_keywords", []))
    except json.JSONDecodeError:
        try:
            entity_kw_result = "{" + entity_kw_result.split("{")[1].split("}")[0] + "}"
            entity_kw_data = json.loads(entity_kw_result)
            entity_keywords = ", ".join(entity_kw_data.get("entity_keywords", []))
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Failed to extract entity keywords: {e}")
    
    # Parse theme keywords
    theme_keywords = ""
    try:
        theme_kw_data = json.loads(theme_kw_result)
        theme_keywords = ", ".join(theme_kw_data.get("theme_keywords", []))
    except json.JSONDecodeError:
        try:
            theme_kw_result = "{" + theme_kw_result.split("{")[1].split("}")[0] + "}"
            theme_kw_data = json.loads(theme_kw_result)
            theme_keywords = ", ".join(theme_kw_data.get("theme_keywords", []))
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Failed to extract theme keywords: {e}")
    
    if not entity_keywords and not theme_keywords:
        logger.error("Failed to extract any keywords")
        return PROMPTS["fail_response"]
    
    # Retrieve from entity and theme hypergraphs
    entity_context = None
    theme_context = None
    
    if entity_keywords:
        entity_context = await _build_entity_query_context(
            entity_keywords,
            knowledge_hypergraph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    
    if theme_keywords:
        theme_context = await _build_relation_query_context(
            theme_keywords,
            knowledge_hypergraph_theme,
            keys_vdb,
            themes_vdb,
            text_chunks_db,
            query_param,
        )
    
    # Merge contexts
    context = combine_contexts(theme_context, entity_context)
    
    if query_param.only_need_context:
        return context
    if context is None:
        logger.warning("No context retrieved from either hypergraph")
        return PROMPTS["fail_response"]
    
    # Generate response
    sys_prompt = PROMPTS["rag_response"].format(
        context_data=context, response_type=query_param.response_type
    )
    define_str = PROMPTS["rag_define"].format(
        entity_keywords=entity_keywords or "", theme_keywords=theme_keywords or ""
    )
    response = await use_model_func(query + define_str, system_prompt=sys_prompt)
    
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    return response


def combine_contexts(
    relation_context: Optional[str],
    entity_context: Optional[str],
) -> str:
    """Combine and deduplicate contexts from different sources.
    
    Merges entity and relation contexts while removing duplicates
    from entities, relationships, and sources sections.
        
    Returns:
        Combined context string with deduplicated sections.
    """
    def extract_sections(context: str) -> Tuple[str, str, str]:
        """Extract sections from context."""
        entities_match = re.search(
            r"-----Entities-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        relationships_match = re.search(
            r"-----Relationships-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        sources_match = re.search(
            r"-----Sources-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )

        entities = entities_match.group(1) if entities_match else ""
        relationships = relationships_match.group(1) if relationships_match else ""
        sources = sources_match.group(1) if sources_match else ""

        return entities, relationships, sources

    # Extract from theme and entity contexts
    if relation_context is None:
        warnings.warn(
            "Theme context is None. Return empty theme entity/relationship/source"
        )
        hl_entities, hl_relationships, hl_sources = "", "", ""
    else:
        hl_entities, hl_relationships, hl_sources = extract_sections(relation_context)

    if entity_context is None:
        warnings.warn(
            "Entity context is None. Return empty entity-level entity/relationship/source"
        )
        ll_entities, ll_relationships, ll_sources = "", "", ""
    else:
        ll_entities, ll_relationships, ll_sources = extract_sections(entity_context)

    # Merge and deduplicate results
    combined_entities = process_combine_contexts(hl_entities, ll_entities)
    combined_relationships = process_combine_contexts(hl_relationships, ll_relationships)
    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    return f"""
-----Entities-----
```csv
{combined_entities}
```
-----Relationships-----
```csv
{combined_relationships}
```
-----Sources-----
```csv
{combined_sources}
```
"""


def remove_after_sources(input_string: str) -> str:
    """Remove content after '-----Sources-----' marker.
    
    Args:
        input_string: Input string to process.
        
    Returns:
        String with content after Sources marker removed.
    """
    index = input_string.find("-----Sources-----")
    if index != -1:
        return input_string[:index]
    return input_string


async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """Execute a naive vector similarity query.
    
    This simple query mode performs direct vector similarity search
    on text chunks without using the hypergraph structure.
        
    Returns:
        Generated response string or fail response if no results found.
    """
    use_model_func = global_config["llm_model_func"]
    
    # Query vector database
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        logger.warning("No chunks retrieved from vector database")
        return PROMPTS["fail_response"]
    
    # Retrieve actual chunk content from text_chunks_db
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)
    
    # Limit by token size and format context
    truncated_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    logger.info(f"Truncated {len(chunks)} to {len(truncated_chunks)} chunks")
    
    # Format context string
    context = "--New Chunk--\n".join([c["content"] for c in truncated_chunks])
    
    if query_param.only_need_context:
        return context
    
    if not context:
        logger.warning("No context available after truncation")
        return PROMPTS["fail_response"]
    
    # Generate response from context
    sys_prompt = PROMPTS["naive_rag_response"].format(
        content_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(query, system_prompt=sys_prompt)
    
    if len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )
    
    return response
