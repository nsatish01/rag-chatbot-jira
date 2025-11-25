import ollama
from retrieval import search

SYSTEM_PROMPT = """
You are a helpful assistant that answers questions based ONLY on the provided Jira ticket context.
If the answer is not in the context, say: 'The information is not available in the Jira data.'
Be precise and cite the ticket IDs where appropriate.
"""


def build_prompt(query, retrieved_chunks):
    context_text = ""

    for chunk in retrieved_chunks:
        ticket_id = chunk.get("ticket_id", "Unknown")
        text = chunk["chunk_text"]
        context_text += f"[Ticket {ticket_id}] {text}\n\n"

    prompt = f"""
{SYSTEM_PROMPT}

Context from Jira tickets:
--------------------------------
{context_text}
--------------------------------

User question: {query}
