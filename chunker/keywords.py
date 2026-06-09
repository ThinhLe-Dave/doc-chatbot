from __future__ import annotations

import re
from collections import Counter
from typing import List

_KEYWORD_RE = re.compile(r"[a-z][a-z0-9]+")
_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "what", "which", "who",
    "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "because", "but",
    "if", "while", "about", "between", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "also", "into", "many",
}


def extract_keywords(text: str, top_n: int = 3) -> List[str]:
    tokens = [
        word
        for word in _KEYWORD_RE.findall(text.lower())
        if word not in _STOP_WORDS and len(word) > 2
    ]
    if not tokens:
        return []
    return [word for word, _ in Counter(tokens).most_common(top_n)]
