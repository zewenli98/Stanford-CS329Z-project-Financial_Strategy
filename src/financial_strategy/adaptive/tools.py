"""A closed, point-in-time tool surface with opaque model-visible identities.

Offsets always refer to the redacted document returned by this surface, never to
an unrelated raw file. Private source records remain available to the audit writer.
Masking removes direct identifiers; contextual re-identification remains possible.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from financial_strategy.errors import StrategyError

_AMBIGUOUS_COMPANY_NAMES = frozenset(
    {"target", "gap", "block", "carrier", "progressive", "southern", "united"}
)
_COMPANY_CONTEXT = (
    r"(?=\s+(?i:shares?|stock|earnings|revenue|retailer|reported|reports|announced|"
    r"announces|raised|raises|cut|cuts|reaffirmed|said|says)\b)"
)


def _company_name_pattern(name: str) -> re.Pattern:
    """Keep ordinary prose distinct from ambiguous, shortened issuer names."""
    boundary = r"(?<![\w])"
    ending = r"(?![\w])"
    if name.casefold() not in _AMBIGUOUS_COMPANY_NAMES:
        return re.compile(boundary + re.escape(name) + ending, re.IGNORECASE)
    # A bare name needs a capital within that exact token, not in a later word.
    capital = rf"(?=.{{0,{len(name) - 1}}}[A-Z])"
    return re.compile(
        boundary + capital + "(?i:" + re.escape(name) + ")" + ending + _COMPANY_CONTEXT
    )


class ResearchToolError(StrategyError):
    """An invalid action cannot expand the worker's permitted data boundary."""


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


class OpaqueToolView:
    """Expose only frozen eligible evidence through five bounded local operations."""

    TOOL_NAMES = frozenset({"search", "read", "features", "timeline", "cards"})

    def __init__(
        self,
        snapshot: Any,
        task: Any,
        universe: dict,
        features: Mapping[str, Any],
        alias_map: Mapping[str, str] | None = None,
        *,
        company_root: str | Path | None = None,
        macro_root: str | Path | None = None,
        reserve_tool: Callable[[str], None] | None = None,
        memory: Any = None,
        max_read_chars: int = 4000,
        protocol_v2: bool = False,
    ) -> None:
        self.snapshot = snapshot
        self.task = task
        self.cutoff = task.decision_cutoff
        if self.cutoff.tzinfo is None:
            raise ValueError("tool cutoff must be timezone-aware")
        securities = universe.get("securities", [])
        self._securities = {item["symbol"]: item for item in securities}
        eligible = sorted(set(self._securities) & set(features))
        self.aliases = (
            {symbol: alias_map[symbol] for symbol in eligible}
            if alias_map is not None
            else {
                symbol: f"company_{sorted(self._securities).index(symbol) + 1:04d}"
                for symbol in eligible
            }
        )
        if len(set(self.aliases.values())) != len(self.aliases):
            raise ValueError("company aliases must be unique")
        if any(not re.fullmatch(r"company_[a-zA-Z0-9]+", alias) for alias in self.aliases.values()):
            raise ValueError("company aliases must use opaque company_ identifiers")
        self.reverse_aliases = {alias: symbol for symbol, alias in self.aliases.items()}
        self.features = {symbol: features[symbol] for symbol in eligible}
        self.reserve_tool = reserve_tool
        self.memory = memory
        self.max_read_chars = max(1, min(int(max_read_chars), 12000))
        self.protocol_v2 = protocol_v2
        self._roots = {
            "company": Path(company_root).resolve() if company_root is not None else None,
            "macro": Path(macro_root).resolve() if macro_root is not None else None,
        }
        self._replacements: list[tuple[re.Pattern, str]] = []
        # Include out-of-universe company names in redaction rules: a mention of
        # another known company must not reveal a selected company's identity.
        for symbol, security in self._securities.items():
            alias = self.aliases.get(symbol, "[other_company]")
            identifiers = {
                symbol,
                str(security.get("cik", "")),
                str(security.get("permanent_entity_id", "")),
            }
            company_names: set[str] = set()
            name = str(security.get("company_name", "")).strip()
            if name:
                company_names.add(name)
                short = re.sub(
                    r",?\s+(?:INCORPORATED|CORPORATION|CORP|INC|CO|LTD|PLC)\.?\b.*$",
                    "",
                    name,
                    flags=re.IGNORECASE,
                ).strip()
                if len(short) >= 3:
                    company_names.add(short)
            for company_name in company_names:
                self._replacements.append((_company_name_pattern(company_name), alias))
            for identifier in sorted(identifiers, key=len, reverse=True):
                if identifier and identifier != "None":
                    if identifier == symbol:
                        # Common English words and single letters are not safe bare identifiers.
                        if len(symbol) <= 2 or symbol in {"LOW", "ALL", "ARE", "NOW", "IT"}:
                            pattern = re.compile(r"\$" + re.escape(symbol) + r"\b")
                        else:
                            pattern = re.compile(r"(?<![\w])" + re.escape(symbol) + r"(?![\w])")
                        self._replacements.append((pattern, alias))
                        continue
                    pattern = re.compile(
                        r"(?<![\w])" + re.escape(identifier) + r"(?![\w])", re.IGNORECASE
                    )
                    self._replacements.append((pattern, alias))
        self._replacements.sort(key=lambda pair: len(pair[0].pattern), reverse=True)
        self.evidence_map: dict[str, Any] = {}
        self.documents: dict[str, str] = {}
        self._records: dict[str, dict] = {}
        candidates = [*snapshot.company_evidence, *snapshot.macro_evidence]
        candidates.sort(key=lambda item: (item.available_at, item.evidence_id))
        seen_hashes: set[str] = set()
        for record in candidates:
            if record.available_at > self.cutoff:
                continue
            if record.scope == "company" and record.symbol not in self.aliases:
                continue
            if record.content_hash in seen_hashes:
                continue
            seen_hashes.add(record.content_hash)
            source_id = "source_" + hashlib.sha256(record.content_hash.encode()).hexdigest()[:16]
            event_key = json.dumps(
                [
                    record.scope,
                    record.symbol,
                    record.event_type,
                    str(record.event_time or record.published_at),
                ]
            )
            event_id = "event_" + hashlib.sha256(event_key.encode()).hexdigest()[:16]
            document = self._load_document(record)
            self.evidence_map[source_id] = record
            self.documents[source_id] = self.redact_text(document)
            self._records[source_id] = {
                "evidence_id": source_id,
                "event_id": event_id,
                "scope": record.scope,
                "symbol": self.aliases.get(record.symbol),
                "title": self.redact_text(record.title),
                "excerpt": self.redact_text(record.excerpt),
                "available_at": record.available_at.isoformat(),
                "event_time": record.event_time.isoformat() if record.event_time else None,
                "event_type": record.event_type,
                "event_types": record.event_types,
                "event_status": getattr(record.event_status, "value", record.event_status),
                "source_type": record.source_type,
                "document_length": len(self.documents[source_id]),
            }
            if record.scope == "macro":
                self._records[source_id].update(
                    {
                        key: getattr(record, key, None)
                        for key in (
                            "actual",
                            "consensus",
                            "unit",
                            "affected_risk_factors",
                            "consensus_published_at",
                        )
                    }
                )
                self._records[source_id] = self.sanitize_model(self._records[source_id])
        self.trace: list[dict] = []

    def _load_document(self, record: Any) -> str:
        root = self._roots[record.scope]
        if root is not None and record.content_path:
            candidate = (root / record.content_path).resolve()
            if not candidate.is_relative_to(root):
                raise ResearchToolError("UNSAFE_DOCUMENT_PATH", "document leaves frozen root")
            if candidate.is_file():
                # The frozen record excerpt is itself an eligible source input,
                # and often carries the useful facts before long site navigation.
                # Label it explicitly; offsets refer to this reproducible view.
                return (
                    f"Frozen evidence record excerpt:\n{record.title}\n{record.excerpt}"
                    f"\n\nFrozen source document:\n{candidate.read_text(encoding='utf-8')}"
                )
        return f"{record.title}\n{record.excerpt}"

    def redact_text(self, text: str) -> str:
        # No URL/path/source accession identifier is useful to the model: citations
        # use opaque IDs and are resolved privately during deterministic checks.
        redacted = re.sub(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", "[source_url]", str(text))
        redacted = re.sub(r"\b[\w.-]+\.(?:com|org|net|gov|io)\b", "[source_domain]", redacted)
        redacted = re.sub(r"[A-Za-z]:[\\/][^\s\"<>]+", "[source_path]", redacted)
        redacted = re.sub(
            r"(?:\b[\w.-]+/)+(?:[\w.-]+\.(?:txt|jsonl?|html?|pdf))\b", "[source_path]", redacted
        )
        redacted = re.sub(r"\b\d{10}-\d{2}-\d{6}\b", "[filing_id]", redacted)
        redacted = re.sub(
            r"\b(?:CIK|sec_cik_)\s*[:#]?\s*\d+\b", "[entity_id]", redacted, flags=re.IGNORECASE
        )
        for pattern, alias in self._replacements:
            redacted = pattern.sub(alias, redacted)
        return redacted

    def sanitize_model(self, value: Any) -> Any:
        """Apply identity masking recursively, including string dictionary keys."""
        value = _plain(value)
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, dict):
            return {
                self.redact_text(str(key)): self.sanitize_model(item) for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.sanitize_model(item) for item in value]
        return value

    def private_mapping(self) -> dict:
        return {
            "companies": {alias: symbol for symbol, alias in self.aliases.items()},
            "sources": {
                opaque: {
                    "evidence_id": item.evidence_id,
                    "symbol": item.symbol,
                    "source_url": str(item.source_url) if item.source_url else None,
                    "content_path": item.content_path,
                    "content_hash": item.content_hash,
                }
                for opaque, item in self.evidence_map.items()
            },
        }

    def resolve_private(self, alias: str) -> str:
        try:
            return self.reverse_aliases[alias]
        except KeyError as exc:
            raise ResearchToolError("UNKNOWN_COMPANY", "unknown opaque company") from exc

    def feature_payload(self, alias: str) -> dict:
        symbol = self.resolve_private(alias)
        raw = _plain(self.features[symbol])
        keep = {
            key: value
            for key, value in raw.items()
            if key
            not in {
                "feature_id",
                "symbol",
                "snapshot_id",
                "input_sessions",
                "input_columns",
                "formula_version",
            }
        }
        historical = keep.get("historical_context")
        if isinstance(historical, dict):
            keep["historical_context"] = {
                key: value
                for key, value in historical.items()
                if key
                not in {
                    "feature_id",
                    "symbol",
                    "snapshot_id",
                    "input_sessions",
                    "input_columns",
                    "formula_version",
                }
            }
        return self.sanitize_model({"symbol": alias, **keep})

    def model_context(self) -> dict:
        exposures = {
            item.symbol: item.exposures for item in getattr(self.snapshot, "exposures", [])
        }
        return {
            "decision_cutoff": self.cutoff.isoformat(),
            "target_session": self.task.target_session.isoformat(),
            "range_definition": "regular-session (high-low)/low, in ratio units",
            "companies": [
                {
                    "symbol": alias,
                    "sector": self._securities[symbol].get("sector", "unknown"),
                    "macro_exposures": exposures.get(symbol, {}),
                    "features": self.feature_payload(alias),
                }
                for symbol, alias in self.aliases.items()
            ],
            "tools": sorted(self.TOOL_NAMES),
        }

    def lead_index(self) -> list[dict]:
        """Return company filings plus a bounded macro lead set.

        Daily rates and VIX stay searchable in full. The planner only receives the
        latest print of each macro series and prints inside the news lookback, so
        three years of daily observations do not overflow the request limit.
        """

        records = [dict(item) for item in self._records.values()]
        company = [item for item in records if item.get("scope") != "macro"]
        macro = [item for item in records if item.get("scope") == "macro"]
        if not macro:
            return company
        lookback_days = int(getattr(self.task, "news_lookback_days", 7))
        window_start = self.cutoff - timedelta(days=lookback_days)

        def stamp(item: dict) -> datetime:
            parsed = datetime.fromisoformat(item["available_at"])
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=self.cutoff.tzinfo)
            return parsed

        latest: dict[str, dict] = {}
        for item in macro:
            event_type = str(item.get("event_type") or "")
            current = latest.get(event_type)
            if current is None or stamp(item) > stamp(current):
                latest[event_type] = item
        selected = {
            item["evidence_id"]: item
            for item in latest.values()
            if stamp(item) <= self.cutoff
        }
        for item in macro:
            seen = stamp(item)
            if window_start <= seen <= self.cutoff:
                selected[item["evidence_id"]] = item
        ordered = sorted(selected.values(), key=lambda item: (stamp(item), item["evidence_id"]))
        return [*company, *ordered]

    def source_span(self, evidence_id: str, start: int, end: int) -> dict:
        if evidence_id not in self.documents:
            raise ResearchToolError("UNKNOWN_EVIDENCE", "unknown opaque evidence")
        text = self.documents[evidence_id]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(text)
        ):
            raise ResearchToolError("INVALID_SOURCE_SPAN", "source span bounds are invalid")
        return {"evidence_id": evidence_id, "start": start, "end": end, "quote": text[start:end]}

    def dispatch(self, action: Any) -> dict:
        raw = _plain(action)
        name = raw.get("tool")
        args = raw.get("arguments", {})
        if name not in self.TOOL_NAMES or not isinstance(args, dict):
            raise ResearchToolError("TOOL_NOT_ALLOWED", "tool is outside the local allowlist")
        if self.reserve_tool is not None:
            self.reserve_tool(name)
        result = self._execute(name, args)
        self.trace.append(
            {
                "tool": name,
                "arguments": args,
                "result_hash": hashlib.sha256(
                    json.dumps(result, sort_keys=True).encode()
                ).hexdigest(),
            }
        )
        return result

    def _execute(self, name: str, args: dict) -> dict:
        allowed = {
            "search": {"query", "symbol", "scope", "top_n"},
            "read": {"evidence_id", "start", "end"},
            "features": {"symbol"},
            "timeline": {"symbol"},
            "cards": {"symbol"},
        }
        if set(args) - allowed[name]:
            raise ResearchToolError("INVALID_TOOL_ARGUMENT", "unexpected tool argument")
        alias = args.get("symbol")
        if alias is not None and alias not in self.reverse_aliases:
            raise ResearchToolError("UNKNOWN_COMPANY", "unknown opaque company")
        if name == "features":
            return self.feature_payload(alias)
        if name == "read":
            source = args.get("evidence_id")
            start = args.get("start", 0)
            if self.protocol_v2 and "start" not in args and source in self.documents:
                start = self.preferred_start(source)
            if (
                source not in self.documents
                or not isinstance(start, int)
                or isinstance(start, bool)
            ):
                raise ResearchToolError("INVALID_SOURCE_SPAN", "unknown source or invalid start")
            end = args.get(
                "end", min(len(self.documents[source]), start + min(1200, self.max_read_chars))
            )
            if isinstance(end, int) and end - start > self.max_read_chars:
                raise ResearchToolError("READ_LIMIT_EXCEEDED", "source span exceeds read limit")
            span = self.source_span(source, start, end)
            return {**self._records[source], "span": span}
        if name == "cards":
            cards = self.memory.active(self.cutoff) if self.memory is not None else []
            return {
                "cards": [
                    _plain(card)
                    for card in cards
                    if (alias is None or alias in card.symbols)
                    and all(span.evidence_id in self.evidence_map for span in card.source_spans)
                ]
            }
        records = [
            item
            for item in self._records.values()
            if alias is None or item["symbol"] in {None, alias}
        ]
        if name == "timeline":
            records.sort(key=lambda item: (item["available_at"], item["evidence_id"]))
            return {"events": records}
        scope = args.get("scope")
        if scope not in {None, "company", "macro"}:
            raise ResearchToolError("INVALID_TOOL_ARGUMENT", "unknown evidence scope")
        records = [item for item in records if scope is None or item["scope"] == scope]
        query = args.get("query", "")
        if not isinstance(query, str) or len(query) > 1000:
            raise ResearchToolError("INVALID_TOOL_ARGUMENT", "query must be bounded text")
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        top_n = args.get("top_n", 8)
        if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 20:
            raise ResearchToolError("INVALID_TOOL_ARGUMENT", "top_n must be between 1 and 20")
        ranked = []
        for item in records:
            text = f"{item['title']}\n{self.documents[item['evidence_id']]}".lower()
            score = sum(text.count(term) for term in terms)
            if not terms or score:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["evidence_id"]))
        return {
            "hits": [{**item, "match_count": score} for score, item in ranked[:top_n]],
            "eligible_count": len(records),
        }

    def preferred_start(self, source):
        text = self.documents[source]
        # Press releases are useful near the beginning; filings often start with XBRL.
        if re.search(r"EX-99|PRESS RELEASE|FOR IMMEDIATE RELEASE", text[:1000], re.I):
            return 0
        matches = list(
            re.finditer(
                r"Item\s+(?:2\.02|5\.02|8\.01|3\.02)|Results of Operations and Financial Condition",
                text,
                re.I,
            )
        )
        if matches:
            return max(0, matches[-1].start() - 100)
        return 0

    def packet_observations(self, alias):
        records = [r for r in self._records.values() if r["symbol"] == alias]
        # Prefer substantive exhibits and recent announcements over 10-K boilerplate.
        records.sort(
            key=lambda r: (
                -(datetime.fromisoformat(r["available_at"]).timestamp()),
                0 if "EX-99" in self.documents[r["evidence_id"]][:500] else 1,
                r["evidence_id"],
            )
        )
        exhibits = [r for r in records if "EX-99" in self.documents[r["evidence_id"]][:500]]
        chosen = (exhibits + [r for r in records if r not in exhibits])[:2]
        observations = []
        for record in chosen:
            source = record["evidence_id"]
            text = self.documents[source]
            starts = [self.preferred_start(source)]
            # Include explicit counterevidence/outlook passages instead of only the headline.
            match = re.search(
                r"(?:not changed|unchanged|outlook|guidance|expects|expect to)", text, re.I
            )
            if match and match.start() > starts[0] + 1400:
                starts.append(max(0, match.start() - 250))
            elif len(text) > starts[0] + 1800:
                starts.append(starts[0] + 1800)
            for start in starts[:2]:
                arguments = {
                    "evidence_id": source,
                    "start": start,
                    "end": min(len(text), start + 1800),
                }
                output = self.dispatch({"tool": "read", "arguments": arguments})
                observations.append(
                    {"step": -1, "tool": "read", "arguments": arguments, "result": output}
                )
        return observations
