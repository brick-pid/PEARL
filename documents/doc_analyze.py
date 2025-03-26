import os
import json
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")

# Directory containing JSONL files
chunks_dir = "documents/chunks"

output_dir = "documents/analysis"

# Function to create histogram
def create_histogram(token_counts, filename):
    plt.figure(figsize=(10, 6))
    plt.hist(token_counts, bins=50, edgecolor='black')
    plt.title(f'Token Count Distribution for {filename}')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.savefig(f'{output_dir}/{filename.replace(".jsonl", "")}_histogram.png')
    plt.close()


# Iterate through each JSONL file in the chunks directory
for filename in os.listdir(chunks_dir):
    if filename.endswith('.jsonl'):
        file_path = os.path.join(chunks_dir, filename)
        token_counts = []

        print(f"Processing {filename}...")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    chunk = data.get('chunk', '')
                    tokens = tokenizer.encode(chunk)
                    token_count = len(tokens)
                    token_counts.append(token_count)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file {filename}")
                except Exception as e:
                    print(f"Error processing line in file {filename}: {str(e)}")

        # Create histogram for this file
        create_histogram(token_counts, filename)

        # Print statistics for this file
        print(f"Statistics for {filename}:")
        print(f"  Total chunks: {len(token_counts)}")
        print(f"  Min tokens: {min(token_counts)}")
        print(f"  Max tokens: {max(token_counts)}")
        print(f"  Average tokens: {sum(token_counts) / len(token_counts):.2f}")
        print()

print("Processing complete. Histogram images have been saved.")
