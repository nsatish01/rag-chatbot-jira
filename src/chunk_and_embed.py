
from transformers import AutoTokenizer
import pandas as pd

# Load your CSV
jira_df = pd.read_csv("C:\\Users\\snagarajan\\AI_learnings\\rag-chatbot-jira\\data\\sample_jira_tickets.csv")

# Initialize tokenizer for LLaMA 2 (or any HF model you want)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")  # adjust model name as needed

def chunk_text(text, max_tokens=200):
    # Tokenize the text
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        # Decode back to text for each chunk
        chunk = tokenizer.decode(tokens[i:i+max_tokens])
        chunks.append(chunk)
    return chunks

jira_chunks = []
for idx, row in jira_df.iterrows():
    ticket_id = row['TicketID']
    for chunk in chunk_text(row['text'], max_tokens=200):
        jira_chunks.append({"ticket_id": ticket_id, "chunk": chunk})

print("Total chunks:", len(jira_chunks))
