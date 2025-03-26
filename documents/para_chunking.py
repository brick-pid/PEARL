"""
every 30 \n\n is a chunk
"""

"""
Chunk documents into groups of 30 paragraphs each.
"""
import json
import jsonlines
import os
from typing import List


def chunk_document(doc: dict, paragraphs_per_chunk: int = 30, max_chunk_size: int = 20000) -> List[str]:
    """
    Split the document into chunks of specified number of paragraphs or maximum size.
    """
    paragraphs = doc['html'].split('\n\n')
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    
    for paragraph in paragraphs:
        if len(current_chunk) >= paragraphs_per_chunk or current_chunk_size + len(paragraph) > max_chunk_size:
            chunks.append('\n\n'.join(current_chunk).strip())
            current_chunk = []
            current_chunk_size = 0
        
        current_chunk.append(paragraph)
        current_chunk_size += len(paragraph)
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk).strip())
    
    return chunks


def process_json_file(input_file: str, paragraphs_per_chunk: int = 30, max_chunk_size: int = 20000) -> List[dict]:
    """
    Process a single JSON file and chunk the documents based on paragraph count and maximum chunk size.
    
    Args:
        input_file (str): JSON file path to be processed.
        paragraphs_per_chunk (int): Number of paragraphs per chunk.
        max_chunk_size (int): Maximum size of each chunk in characters.
    
    Returns:
        List[dict]: List of chunked data dictionaries.
    """
    chunked_data = []

    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return chunked_data

    # Load JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    for doc in documents:
        chunks = chunk_document(doc, paragraphs_per_chunk, max_chunk_size)
        for chunk in chunks:
            chunk_data = {
                "title": doc['title'],
                "url": doc['url'],
                "chunk": chunk
            }
            chunked_data.append(chunk_data)

    return chunked_data


if __name__ == "__main__":
    input_files = ["documents/docs/julia_manual-1.json", "documents/docs/julia_base-1.json"]  # List your input JSON files here
    output_file = "documents/chunks/julia_chunks.jsonl"
    paragraphs_per_chunk = 40
    max_chunk_size = 20000

    combined_chunks = []
    for input_file in input_files:
        combined_chunks.extend(process_json_file(input_file, paragraphs_per_chunk, max_chunk_size))

    with jsonlines.open(output_file, mode='w') as writer:
        for chunk_data in combined_chunks:
            writer.write(chunk_data)
    
    print(f"Chunked data has been written to {output_file}")
