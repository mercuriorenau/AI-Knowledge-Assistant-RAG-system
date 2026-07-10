"""Simple answer-vs-context grounding checks for offline eval."""

from __future__ import annotations

import re

_EMPTY_PREFIX = "i don't know"
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2}


def is_faithful(
    answer: str,
    context: str,
    *,
    context_empty: bool,
    min_overlap: float = 0.15,
) -> bool:
    """
    Empty-context answers must refuse with the known prefix.
    Otherwise require a minimum fraction of answer tokens to appear in context.
    """
    ans = (answer or "").strip()
    if not ans:
        return False
    if context_empty or not (context or "").strip():
        return ans.lower().startswith(_EMPTY_PREFIX)
    ctx_tokens = _tokens(context)
    ans_tokens = _tokens(ans)
    if not ans_tokens:
        return False
    if not ctx_tokens:
        return False
    overlap = len(ans_tokens & ctx_tokens) / len(ans_tokens)
    return overlap >= min_overlap
