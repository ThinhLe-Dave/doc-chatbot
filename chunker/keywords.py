from __future__ import annotations

import re
from collections import Counter
from typing import List

_KEYWORD_RE = re.compile(r"\w+", re.IGNORECASE)


def extract_keywords(text: str, top_n: int = 3) -> List[str]:
    tokens = [
        word
        for word in _KEYWORD_RE.findall(text.lower())
        if len(word) > 2
    ]
    if not tokens:
        return []
    return [word for word, _ in Counter(tokens).most_common(top_n)]
