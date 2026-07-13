"""In-memory server-side conversation session store.

Keeps per-session chat history so the chatbot can remember prior turns.
Sessions are kept in process memory with a sliding message window and idle
expiry. This intentionally has no external dependencies; for multi-process or
persistent deployments, back this with Redis or a database instead.
"""

from __future__ import annotations

import re
import time
import uuid
from threading import Lock
from typing import Dict, List, Optional

# Keep the last N messages (user+assistant), i.e. MAX_MESSAGES // 2 exchanges.
MAX_MESSAGES = 20
# Drop sessions that have been idle longer than this (seconds).
SESSION_TTL = 60 * 60 * 6  # 6 hours
# Hard cap on the number of concurrent sessions retained in memory.
MAX_SESSIONS = 2000

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

_sessions: Dict[str, dict] = {}
_lock = Lock()


def _now() -> float:
    return time.time()


def _is_valid_id(session_id: object) -> bool:
    return isinstance(session_id, str) and bool(_UUID_RE.match(session_id))


def _evict_locked(now: float) -> None:
    """Remove expired sessions and enforce the session cap. Assumes lock held."""
    expired = [sid for sid, s in _sessions.items() if now - s["last_access"] > SESSION_TTL]
    for sid in expired:
        _sessions.pop(sid, None)

    overflow = len(_sessions) - MAX_SESSIONS
    if overflow > 0:
        oldest = sorted(_sessions, key=lambda sid: _sessions[sid]["last_access"])[:overflow]
        for sid in oldest:
            _sessions.pop(sid, None)


def get_or_create_session(session_id: Optional[str]) -> str:
    """Return a valid session id, creating a new session when needed.

    A client-supplied id is only honored when it is a well-formed UUID that
    already exists in the store. Any other value (missing, malformed, or
    unknown) yields a fresh server-generated id, so clients cannot choose or
    fixate arbitrary session keys or read another session's history.
    """
    now = _now()
    with _lock:
        _evict_locked(now)
        if _is_valid_id(session_id) and session_id in _sessions:
            _sessions[session_id]["last_access"] = now
            return session_id
        new_id = str(uuid.uuid4())
        _sessions[new_id] = {"messages": [], "last_access": now}
        return new_id


def get_history(session_id: str) -> List[Dict[str, str]]:
    """Return a copy of the message history for a session (empty if unknown)."""
    if not _is_valid_id(session_id):
        return []
    with _lock:
        session = _sessions.get(session_id)
        if not session:
            return []
        session["last_access"] = _now()
        return list(session["messages"])


def append_turn(session_id: str, user_message: str, assistant_message: str) -> None:
    """Append a user+assistant exchange, trimming to the sliding window."""
    if not _is_valid_id(session_id):
        return
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            session = {"messages": [], "last_access": _now()}
            _sessions[session_id] = session
        session["messages"].append({"role": "user", "content": user_message})
        session["messages"].append({"role": "assistant", "content": assistant_message})
        if len(session["messages"]) > MAX_MESSAGES:
            session["messages"] = session["messages"][-MAX_MESSAGES:]
        session["last_access"] = _now()


def clear_session(session_id: str) -> None:
    """Forget a session's history entirely."""
    if not _is_valid_id(session_id):
        return
    with _lock:
        _sessions.pop(session_id, None)
