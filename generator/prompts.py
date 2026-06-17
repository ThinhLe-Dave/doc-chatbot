from typing import Dict, List, Optional

NO_CONTEXT = "No relevant context found."
NO_ANSWER = "No relevant information found in the database."

SYSTEM_PROMPT = (
    "You are a document assistant. Answer ONLY from the provided context. "
    "If the context does not contain enough information, say exactly: "
    f"\"{NO_ANSWER}\". Do not use outside knowledge. Be concise. "
    "Cite sources with available book, chapter, verse, section, page, or source path."
)

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {query}

Answer:"""


def build_messages(query: str, context: Optional[str]) -> List[Dict[str, str]]:
    safe_context = " ".join(str(context).split()) if context else NO_CONTEXT
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(context=safe_context or NO_CONTEXT, query=query)},
    ]
