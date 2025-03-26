"""
chunk whole document into semantic chunks;
input:
    json file, like:
    [
        {
            "title": "Unicode Input · The Julia Language",
            "url": "https://docs.julialang.org/en/v1/manual/unicode-input/",
            "html": "..."
        },
        ...
    ]
process: 
    split the document into chunks by the smallest title;
output:
jsonl file of chunks, like:
    {
        "title": "...",
        "url": "...",
        "chunk": "..."
    }
"""
import json
import jsonlines
import re
import os
from typing import List


def chunk_document(doc, splitter_pattern):
    """
    Split the document into chunks using the given splitter pattern, preserving the titles.
    """
    matches = list(re.finditer(splitter_pattern, doc['html']))
    chunks = []

    # If no matches found, return the entire document as a single chunk
    if not matches:
        return [doc['html'].strip()]

    # Get all start and end positions of matches
    indices = [(match.start(), match.end()) for match in matches]

    # Iterate over indices to create chunks from one title to the next
    # include the first chunk
    if indices[0][0] > 0:
        chunks.append(doc['html'][0:indices[0][0]])
    for i in range(len(indices)):
        start_idx = indices[i][0]
        end_idx = indices[i + 1][0] if i + 1 < len(indices) else len(doc['html'])
        chunk = doc['html'][start_idx:end_idx]
        chunks.append(chunk)

    return chunks


def process_json_file(input_file: str, splitter_pattern: str):
    """
    Process a single JSON file and chunk the documents based on the smallest title format.
    
    Args:
        input_file (str): JSON file path to be processed.
        splitter_pattern (str): Regex pattern to identify title-based splits.
    
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
        # Split the document into chunks by the smallest title format
        chunks = chunk_document(doc, splitter_pattern)
        for chunk in chunks:
            chunk_data = {
                "title": doc['title'],
                "url": doc['url'],
                "chunk": chunk  # Removing unnecessary whitespace
            }
            chunked_data.append(chunk_data)

    return chunked_data


if __name__ == "__main__":
    # Define input files, splitter pattern, and output file
    input_files = ["documents/docs/lua_manual-1.json"]  # List your input JSON files here
    output_file = "documents/chunks/lua_chunks.jsonl"

    # Updated splitter pattern
    splitter_pattern = r"(?<!\n)\n\n\d{1,2}(\.\d{1,2}){1,3}"

    # Process the JSON files and output the chunks
    combined_chunks = []
    for input_file in input_files:
        combined_chunks.extend(process_json_file(input_file, splitter_pattern))

    with jsonlines.open(output_file, mode='w') as writer:
        for chunk_data in combined_chunks:
            writer.write(chunk_data)
    
    print(f"Chunked data has been written to {output_file}")
