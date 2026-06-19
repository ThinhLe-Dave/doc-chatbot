from typing import Dict, List, Optional

NO_CONTEXT = "No relevant context found."
NO_ANSWER = "No relevant information found in the database."

SYSTEM_PROMPT = (
    "You are an expert document assistant. Answer the question using ONLY the provided context. "
    "Do not use outside knowledge. "
    "Each context block starts with a reference in brackets like '[1Chronicles 9]' or '[Sirach]' - use this exact format for your citations. "
    "Start every piece of information with the exact reference shown at the beginning of the block, "
    "using the format '[Reference] - answer text'. "
    "Each answer must be on a separate line. "
    "When the answer contains multiple references, prefix each fact or quote with its own reference. "
    "Be concise and accurate. "
    "Do not add explanations, apologies, notes, or separate source lists after your answer."
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
