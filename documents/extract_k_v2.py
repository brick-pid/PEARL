"""
Extract knowledge from documents with custom api
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

model_name = "gpt-4o-2024-08-06"

# Initialize OpenAI client
client = openai.OpenAI(
    base_url="https://ssapi.onechat.shop/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

SYSTEM_PROMPT = """You are a highly intelligent and reliable coding assistant, dedicated to delivering accurate and trustworthy information. 
Your task is to analyze the provided {lang} document chunk and extract all relevant programming knowledge. 
You should extract all relevant knowledge from the document, and each piece of knowledge should be informative, self-contained, and focused on a single concept or topic. 

Each piece of knowledge consists of three parts: 

1. Intent: A high-quality, concise explanation of the knowledge. This should reflect the original document context and the extracted knowledge.
2. Knowledge: A high-quality, informative and self-contained description of the extracted knowledge. Describe the knowledge step by step.
3. Code: If the document includes code examples relevant to the knowledge, extract and refine the given code examples to illustrate how this knowledge is applied in practice. Otherwise, you can constuct code examples by yourself.

The JSON...
"""

class KnowledgeItem(BaseModel):
    intent: str
    knowledge: str
    code: str

class KnowledgeList(BaseModel):
    knowledge: List[KnowledgeItem]

def extract_knowledge(document_path, model=model_name):
    """
    use openai model to extract knowledge from the document
    """
    # Read the document
    with open(document_path, 'r', encoding='utf-8') as file:
        document_content = "document to be processed: \n" + file.read()

    # Use OpenAI model to extract knowledge
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": document_content}
        ],
        max_tokens=4096,
        temperature=0.6,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "json_object"}
    )

    try:
        knowledge_list = response.choices[0].message.parsed
        return knowledge_list, None
    except Exception as e:
        return response.choices[0].message.content, str(e)

def main():
    global SYSTEM_PROMPT
    parser = argparse.ArgumentParser(description='Extract knowledge from documents.')
    parser.add_argument('--lang', type=str, choices=['racket', 'julia', 'lua', 'ocaml', 'r'], required=True, help='Programming language to process')
    args = parser.parse_args()
    lang = args.lang

    SYSTEM_PROMPT = SYSTEM_PROMPT.format(lang=lang)
    
    manual_dir = f"documents/documents/manual_{lang}"
    output_dir = f"documents/knowledge/{lang}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    knowledge_file = os.path.join(output_dir, f"{lang}_knowledge_v2.jsonl")
    error_path = os.path.join(output_dir, f"{lang}_errors_v2.jsonl")

    # Get the first document in the directory
    for filename in tqdm(os.listdir(manual_dir), desc="Processing documents"):
        doc_path = os.path.join(manual_dir, filename)

        if doc_path:
            knowledge_list, error = extract_knowledge(doc_path)
            
            # Save extracted knowledge to a new JSONL file
            with open(knowledge_file, 'a', encoding='utf-8') as outfile:
                if isinstance(knowledge_list, KnowledgeList):
                    for item in knowledge_list.knowledge:
                        json.dump(item.model_dump(), outfile)
                        outfile.write('\n')
                else:
                    print(f"Unexpected knowledge format: {type(knowledge_list)}")
                    # also save to the error file
                    with open(error_path, 'a', encoding='utf-8') as error_file:
                        if error:
                            error_file.write(f"{error}\n")
            
            print(f"Extracted knowledge saved to {knowledge_file}")
        else:
            print("No suitable documents found in the specified directory.")

if __name__ == "__main__":
    main()