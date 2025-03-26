"""
Extract knowledge from documents
"""

import os
import openai
from dotenv import load_dotenv
import json
from tqdm import tqdm
from typing import List
from pydantic import BaseModel, Field
import argparse
# Load environment variables
load_dotenv()

model_name = "gpt-4o-mini-2024-07-18"

# Initialize OpenAI client
client = openai.OpenAI(
    base_url="https://ssapi.onechat.shop/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

SYSTEM_PROMPT = """You are an expert in programming language documentation. 
Given a document in {lang}, split it into distinct pieces and extract key knowledge from each. 
Ensure each piece of knowledge should be informative and self-contained, and focused on a single topic. 

Each piece of knowledge consists of four parts: 
1. "content": A concise explanation of one specific feature, syntax, or API usage from the document.
2. "code_demo": Accompanying code example, if available.
3. "knowledge_entity": Keywords describing the main topic of this knowledge. Keywords should follow hierarchy structure, from general to specific keywords, use comma to separate different keywords.
4. "intent": The purpose or typical use case for this knowledge.
"""

class KnowledgeItem(BaseModel):
    content: str
    code_demo: str
    knowledge_entity: str
    intent: str

class KnowledgeList(BaseModel):
    knowledge: List[KnowledgeItem]

def extract_knowledge(doc_chunk, model=model_name, error_file=None):
    """
    use openai model to extract knowledge from the document
    """

    doc_chunk = "document to be processed: \n" + doc_chunk

    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": doc_chunk}
            ],
            max_tokens=8192,
            temperature=0.6,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format=KnowledgeList
        )

        knowledge_list = response.choices[0].message.parsed
        return knowledge_list, None
    except Exception as e:
        error_message = f"Error processing chunk: {str(e)}\n"
        if error_file:
            with open(error_file, 'a', encoding='utf-8') as f:
                f.write(error_message)
                f.write(f"## Problematic doc chunk:\n{doc_chunk}\n\n")
        return None, error_message

def main():
    global SYSTEM_PROMPT    
    chunks_dir = f"documents/chunks/"

    # Get the first document in the directory
    for filename in tqdm(os.listdir(chunks_dir), desc="Processing documents"):
        lang = filename.split('_')[0]
        
        SYSTEM_PROMPT = SYSTEM_PROMPT.format(lang=lang)
        output_dir = f"documents/knowledge/{lang}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        knowledge_file = os.path.join(output_dir, f"{lang}_knowledge_v3_mini.jsonl")
        error_path = os.path.join(output_dir, f"{lang}_errors_v3_mini.jsonl")
        doc_path = os.path.join(chunks_dir, filename)

        # Count the total number of lines in the file
        total_lines = sum(1 for _ in open(doc_path, 'r', encoding='utf-8'))

        # Create a progress bar
        with tqdm(total=total_lines, desc="Processing lines", unit="line") as pbar:
            for line in open(doc_path, 'r', encoding='utf-8'):
                j = json.loads(line)
                doc_chunk = j['url'] + '\n' + j['title'] + '\n' + j['chunk']

                knowledge_list, error = extract_knowledge(doc_chunk, error_file=error_path)
                
                # Save extracted knowledge to a new JSONL file
                with open(knowledge_file, 'a', encoding='utf-8') as outfile:
                    if isinstance(knowledge_list, KnowledgeList):
                        for item in knowledge_list.knowledge:
                            json.dump(item.model_dump(), outfile)
                            outfile.write('\n')
                    else:
                        print(f"## Unexpected knowledge format:\n{type(knowledge_list)}")
                        # also save to the error file
                        with open(error_path, 'a', encoding='utf-8') as error_file:
                            if error:
                                error_file.write(f"{error}\n")
                
                pbar.update(1)  # Update progress bar

        print(f"Extracted knowledge saved to {knowledge_file}")


if __name__ == "__main__":
    main()
