"""Guardrails for LLM-generated SQL (used from Module 7 onward).

Defence in depth. This validator is best-effort; the real safety net is that SqlRetriever
runs every query in a READ ONLY transaction with a statement timeout.

Rules: one statement; SELECT/WITH only; no comments; no quoted identifiers; only whitelisted
tables; no write/DDL/admin keywords; no system catalogs or dangerous functions; row cap.
"""
from __future__ import annotations

import re

from finsight.core.exceptions import UnsafeQueryError

ALLOWED_TABLES = frozenset({"finsight.xbrl_facts", "xbrl_facts"})

_LITERAL = re.compile(r"'(?:[^']|'')*'")
_FORBIDDEN_KW = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|call|do|execute|"
    r"merge|vacuum|analyze|set|reset|listen|notify|lock|into)\b"
)
_FORBIDDEN_NAMES = re.compile(
    r"\b(pg_\w+|dblink\w*|lo_\w+|information_schema|current_setting|set_config)\b"
)
_FROM_JOIN = re.compile(r"\b(?:from|join)\s+([a-z_][a-z0-9_.]*)")
_COMMA_JOIN = re.compile(
    r"\b(?:from|join)\s+[a-z_][a-z0-9_.]*(?:\s+(?:as\s+)?[a-z_]\w*)?\s*,\s*([a-z_][a-z0-9_.]*)"
)
_CTE_NAMES = re.compile(r"(?:\bwith|,)\s*([a-z_]\w*)\s+as\s*\(")
_LIMIT = re.compile(r"\blimit\s+(\d+)")


def validate_sql(sql: str, max_rows: int = 200) -> str:
    """Return a safe, normalised SQL string or raise UnsafeQueryError."""
    s = sql.strip()
    if s.endswith(";"):
        s = s[:-1].rstrip()
    if not s:
        raise UnsafeQueryError("Empty query")

    cleaned = _LITERAL.sub("''", s).lower()  # literals blanked so their contents can't fool us

    if ";" in cleaned:
        raise UnsafeQueryError("Multiple statements are not allowed")
    if "--" in cleaned or "/*" in cleaned:
        raise UnsafeQueryError("SQL comments are not allowed")
    if '"' in cleaned or "$$" in cleaned or "\\" in cleaned:
        raise UnsafeQueryError("Quoted identifiers, dollar quoting and backslashes are not allowed")
    if not re.match(r"\s*(select|with)\b", cleaned):
        raise UnsafeQueryError("Only SELECT queries are allowed")
    if (m := _FORBIDDEN_KW.search(cleaned)):
        raise UnsafeQueryError(f"Forbidden keyword: {m.group(1)}")
    if (m := _FORBIDDEN_NAMES.search(cleaned)):
        raise UnsafeQueryError(f"Forbidden identifier: {m.group(1)}")

    ctes = set(_CTE_NAMES.findall(cleaned))
    referenced = set(_FROM_JOIN.findall(cleaned)) | set(_COMMA_JOIN.findall(cleaned))
    for table in referenced:
        if table not in ALLOWED_TABLES and table not in ctes:
            raise UnsafeQueryError(f"Table not allowed: {table}")

    if (m := _LIMIT.search(cleaned)):
        if int(m.group(1)) > max_rows:
            raise UnsafeQueryError(f"LIMIT exceeds {max_rows}")
    else:
        s = f"{s} LIMIT {max_rows}"
    return s