# -*- coding: utf-8 -*-
"""Utility functions: token encoding, hashing, file I/O, text processing."""

import asyncio
import html
import io
import csv
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union, List, Callable, Optional

import numpy as np
import tiktoken


# Global encoder instance for token operations
ENCODER: Optional[tiktoken.Encoding] = None

# Logger for Cog-RAG operations
logger = logging.getLogger("cograg")


def set_logger(log_file: str) -> None:
    """Configure logger with file handler."""
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)


@dataclass
class EmbeddingFunc:
    """Wrapper for async embedding function with metadata."""
    embedding_dim: int
    max_token_size: int
    func: Callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        """Call the wrapped embedding function."""
        return await self.func(*args, **kwargs)


def locate_json_string_body_from_string(content: str) -> Optional[str]:
    """Extract first JSON object pattern from text."""
    maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
    if maybe_json_str is not None:
        return maybe_json_str.group(0)
    return None


def convert_response_to_json(response: str) -> dict:
    """Extract and parse JSON from LLM response."""
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}")
        raise e from None


def compute_args_hash(*args) -> str:
    """Compute MD5 hash for caching."""
    return md5(str(args).encode()).hexdigest()


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Generate unique ID with MD5 hash and optional prefix."""
    return prefix + md5(content.encode()).hexdigest()


def limit_async_func_call(max_size: int, waiting_time: float = 0.0001):
    """Decorator to limit concurrent async calls without Semaphore."""
    def final_decro(func):
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waiting_time)
            __current_size += 1
            try:
                result = await func(*args, **kwargs)
            finally:
                __current_size -= 1
            return result

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs) -> Callable:
    """Decorator to wrap function as EmbeddingFunc."""
    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


def load_json(file_name: str) -> Optional[dict]:
    """Load JSON from a file.
    
    Args:
        file_name: Path to the JSON file.
        
    Returns:
        Parsed JSON as dictionary, or None if file doesn't exist.
    """
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj: Any, file_name: str) -> None:
    """Write JSON to a file.
    
    Args:
        json_obj: Object to serialize to JSON.
        file_name: Path to the output file.
    """
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o") -> List[int]:
    """Encode a string to tokens using tiktoken.
    
    Args:
        content: Text to encode.
        model_name: Model name for the tokenizer.
        
    Returns:
        List of token IDs.
    """
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    return ENCODER.encode(content)


def decode_tokens_by_tiktoken(tokens: List[int], model_name: str = "gpt-4o") -> str:
    """Decode tokens to a string using tiktoken.
    
    Args:
        tokens: List of token IDs.
        model_name: Model name for the tokenizer.
        
    Returns:
        Decoded string.
    """
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    return ENCODER.decode(tokens)


def pack_user_ass_to_openai_messages(*args: str) -> List[dict]:
    """Pack user/assistant messages for OpenAI API format."""
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} 
        for i, content in enumerate(args)
    ]


def split_string_by_multi_markers(content: str, markers: List[str]) -> List[str]:
    """Split string by multiple markers and return stripped non-empty parts."""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def clean_str(input_val: Any) -> str:
    """Clean string by removing HTML escapes and control characters."""
    if not isinstance(input_val, str):
        return input_val

    result = html.unescape(input_val.strip())
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def is_float_regex(value: str) -> bool:
    """Check if string represents a float number."""
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def truncate_list_by_token_size(
    list_data: List, key: Callable, max_token_size: int
) -> List:
    """Truncate list to fit within token budget."""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def list_of_list_to_csv(data: List[List[str]]) -> str:
    """Convert 2D list to CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(data)
    return output.getvalue()


def csv_string_to_list(csv_string: str) -> List[List[str]]:
    """Parse CSV string to 2D list."""
    output = io.StringIO(csv_string)
    reader = csv.reader(output)
    return [row for row in reader]


def save_data_to_file(data: Any, file_name: str) -> None:
    """Save data to a JSON file.
    
    Args:
        data: Data to save.
        file_name: Path to the output file.
    """
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def xml_to_json(xml_file: str) -> Optional[dict]:
    """Parse GraphML XML file to JSON format.
    
    Extracts nodes and edges from a GraphML file format.
    
    Args:
        xml_file: Path to the GraphML file.
        
    Returns:
        Dictionary with 'nodes' and 'edges' lists, or None on error.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        logger.debug(f"Root element: {root.tag}")
        logger.debug(f"Root attributes: {root.attrib}")

        data = {"nodes": [], "edges": []}
        namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

        for node in root.findall(".//node", namespace):
            node_data = {
                "id": node.get("id").strip('"'),
                "entity_type": (
                    node.find("./data[@key='d0']", namespace).text.strip('"')
                    if node.find("./data[@key='d0']", namespace) is not None
                    else ""
                ),
                "description": (
                    node.find("./data[@key='d1']", namespace).text
                    if node.find("./data[@key='d1']", namespace) is not None
                    else ""
                ),
                "source_id": (
                    node.find("./data[@key='d2']", namespace).text
                    if node.find("./data[@key='d2']", namespace) is not None
                    else ""
                ),
            }
            data["nodes"].append(node_data)

        for edge in root.findall(".//edge", namespace):
            edge_data = {
                "source": edge.get("source").strip('"'),
                "target": edge.get("target").strip('"'),
                "weight": (
                    float(edge.find("./data[@key='d3']", namespace).text)
                    if edge.find("./data[@key='d3']", namespace) is not None
                    else 0.0
                ),
                "description": (
                    edge.find("./data[@key='d4']", namespace).text
                    if edge.find("./data[@key='d4']", namespace) is not None
                    else ""
                ),
                "keywords": (
                    edge.find("./data[@key='d5']", namespace).text
                    if edge.find("./data[@key='d5']", namespace) is not None
                    else ""
                ),
                "source_id": (
                    edge.find("./data[@key='d6']", namespace).text
                    if edge.find("./data[@key='d6']", namespace) is not None
                    else ""
                ),
            }
            data["edges"].append(edge_data)

        logger.info(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")
        return data
    except ET.ParseError as e:
        logger.error(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None


def process_combine_contexts(hl: str, ll: str) -> str:
    """Combine and deduplicate high-level and low-level contexts."""
    header = None
    list_hl = csv_string_to_list(hl.strip())
    list_ll = csv_string_to_list(ll.strip())

    if list_hl:
        header = list_hl[0]
        list_hl = list_hl[1:]
    if list_ll:
        header = list_ll[0]
        list_ll = list_ll[1:]
    if header is None:
        return ""

    if list_hl:
        list_hl = [",".join(item[1:]) for item in list_hl if item]
    if list_ll:
        list_ll = [",".join(item[1:]) for item in list_ll if item]

    combined_sources_set = set(filter(None, list_hl + list_ll))

    combined_sources = [",\t".join(header)]

    for i, item in enumerate(combined_sources_set, start=1):
        combined_sources.append(f"{i},\t{item}")

    return "\n".join(combined_sources)


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """Ensure an event loop is available.
    
    Gets the current event loop or creates a new one if needed.
    
    Returns:
        The current or newly created event loop.
    """
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop