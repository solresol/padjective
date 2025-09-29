"""Utilities for classifying products into WordNet synsets.

This module provides a small workflow that reads product information from a CSV
file, asks an OpenAI model for the best matching WordNet synset (using tool
calling for a structured response), and stores the results in a SQLite database.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

try:  # pragma: no cover - optional dependency for type checking
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class ProductRecord:
    """A single product entry read from the CSV file."""

    product_id: int
    title: str
    tags: str

    @property
    def description(self) -> str:
        """Return a human readable description for prompting the model."""

        if self.tags:
            return f"Title: {self.title}\nTags: {self.tags}"
        return f"Title: {self.title}"


class CSVProductSource:
    """Simple wrapper class that loads product data from a CSV file."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path

    def iter_products(self) -> Iterator[ProductRecord]:
        """Yield :class:`ProductRecord` instances for each product in the CSV."""

        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader, start=1):
                title = (row.get("title") or "").strip()
                tags = (row.get("tags") or "").strip()
                yield ProductRecord(product_id=index, title=title, tags=tags)


@dataclass(slots=True)
class SynsetResult:
    """Structured result returned from the language model."""

    synset_id: Optional[str]
    synset_name: Optional[str]
    synset_definition: Optional[str]
    confidence: Optional[float]
    not_found: bool
    reason: Optional[str]
    raw_response: str
    usage_records: list["UsageRecord"] = field(default_factory=list)


@dataclass(slots=True)
class UsageRecord:
    """Token usage information for a single model invocation."""

    phase: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    recorded_at: datetime


@dataclass(slots=True)
class UsageWindowStats:
    """Rolling aggregate information about token usage."""

    total_tokens: int
    earliest: datetime


class SynsetDatabase:
    """Convenience wrapper around the SQLite database used for results."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.execute("PRAGMA journal_mode=WAL;")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS product_synsets (
                product_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                tags TEXT,
                synset_id TEXT,
                synset_name TEXT,
                synset_definition TEXT,
                confidence REAL,
                not_found INTEGER NOT NULL,
                reason TEXT,
                model TEXT NOT NULL,
                raw_response TEXT NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.commit()

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS synset_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                phase TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY(product_id) REFERENCES product_synsets(product_id)
            )
            """
        )
        self.connection.commit()

    def processed_product_ids(self) -> set[int]:
        cursor = self.connection.execute("SELECT product_id FROM product_synsets")
        return {row[0] for row in cursor.fetchall()}

    def store_result(self, product: ProductRecord, result: SynsetResult, model: str) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO product_synsets (
                product_id,
                title,
                tags,
                synset_id,
                synset_name,
                synset_definition,
                confidence,
                not_found,
                reason,
                model,
                raw_response
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                product.product_id,
                product.title,
                product.tags,
                result.synset_id,
                result.synset_name,
                result.synset_definition,
                result.confidence,
                int(result.not_found),
                result.reason,
                model,
                result.raw_response,
            ),
        )
        self.connection.commit()

    def store_usage(self, product_id: int, usage: Iterable[UsageRecord]) -> None:
        rows = [
            (
                product_id,
                record.phase,
                record.model,
                record.input_tokens,
                record.output_tokens,
                record.total_tokens,
                record.recorded_at.isoformat(),
            )
            for record in usage
            if record.total_tokens or record.input_tokens or record.output_tokens
        ]
        if not rows:
            return
        self.connection.executemany(
            """
            INSERT INTO synset_usage (
                product_id,
                phase,
                model,
                input_tokens,
                output_tokens,
                total_tokens,
                recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.connection.commit()

    def recent_usage(self, window_seconds: int) -> Optional[UsageWindowStats]:
        if window_seconds <= 0:
            return None
        window_start = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        cursor = self.connection.execute(
            """
            SELECT
                SUM(COALESCE(total_tokens, 0)) AS total_tokens,
                MIN(recorded_at) AS first_record
            FROM synset_usage
            WHERE recorded_at >= ?
            """,
            (window_start.isoformat(),),
        )
        row = cursor.fetchone()
        if not row:
            return None
        total_tokens = int(row[0] or 0)
        first_value = row[1]
        if total_tokens <= 0 or not first_value:
            return None
        earliest = _parse_timestamp(first_value)
        if earliest is None:
            earliest = window_start
        return UsageWindowStats(total_tokens=total_tokens, earliest=earliest)

    def close(self) -> None:
        self.connection.close()


def _build_tool_spec() -> Dict[str, object]:
    """Return the JSON schema used for tool calling."""

    return {
        "type": "function",
        "name": "record_synset",
        "function": {
            "name": "record_synset",
            "description": (
                "Select the most appropriate WordNet synset for a product. "
                "Respond with not_found=true if the concept is absent from WordNet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "synset_id": {
                        "type": "string",
                        "description": "The WordNet synset id (e.g. 'n04379243').",
                    },
                    "synset_name": {
                        "type": "string",
                        "description": "Short lemma or gloss name of the synset.",
                    },
                    "synset_definition": {
                        "type": "string",
                        "description": "Definition of the synset from WordNet.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": (
                            "Confidence between 0 and 1 that the synset is a good match."
                        ),
                    },
                    "not_found": {
                        "type": "boolean",
                        "description": "Set to true if there is no matching synset in WordNet.",
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Explanation of how the synset was chosen or why none was found."
                        ),
                    },
                },
                "required": ["not_found"],
                "additionalProperties": False,
            },
        },
    }


def _parse_tool_arguments(tool_payload: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Return the tool arguments regardless of the field used by the API."""

    for key in ("arguments", "input"):
        if key not in tool_payload:
            continue
        value = tool_payload[key]
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return json.loads(value)
    return None


def _extract_tool_response(response_dict: Dict[str, object]) -> Dict[str, object]:
    """Extract the tool call payload from the response dictionary."""

    output_items = response_dict.get("output")
    if not isinstance(output_items, list):
        raise ValueError("Model response did not include any output items")

    for item in output_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            arguments = _parse_tool_arguments(item)
            if arguments is not None:
                return arguments
        if item_type == "tool_call":
            function = item.get("function", {})
            if not isinstance(function, dict):
                continue
            arguments = _parse_tool_arguments(function)
            if arguments is not None:
                return arguments
        if item_type == "tool_use":
            arguments = _parse_tool_arguments(item)
            if arguments is not None:
                return arguments
        content = item.get("content")
        if isinstance(content, list):
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                content_type = content_item.get("type")
                if content_type == "tool_call":
                    function = content_item.get("function", {})
                    if not isinstance(function, dict):
                        continue
                    arguments = _parse_tool_arguments(function)
                    if arguments is not None:
                        return arguments
                if content_type == "tool_use":
                    arguments = _parse_tool_arguments(content_item)
                    if arguments is not None:
                        return arguments
    try:
        serialized_response = json.dumps(
            response_dict, ensure_ascii=False, indent=2, sort_keys=True
        )
    except (TypeError, ValueError):  # pragma: no cover - fallback for unexpected types
        serialized_response = repr(response_dict)
    raise ValueError(
        "No tool call found in model response. Full response:\n"
        f"{serialized_response}"
    )


def call_synset_model(client: "OpenAI", product: ProductRecord, *, model: str) -> SynsetResult:
    """Run a two-step synset selection workflow for the product."""

    initial_messages = [
        {
            "role": "system",
            "content": (
                "You are a linguistic expert that maps product descriptions to "
                "Princeton WordNet synsets. Provide precise nouns and be explicit "
                "when a product does not exist in WordNet."
            ),
        },
        {
            "role": "user",
            "content": (
                "Determine the WordNet synset that best represents this product.\n"
                f"{product.description}\n"
                "Return the result using the record_synset function."
            ),
        },
    ]

    phase_one_result, phase_one_dict, phase_one_usage = _invoke_synset_phase(
        client,
        messages=initial_messages,
        model=model,
        phase="initial",
    )
    usage_records = [phase_one_usage] if phase_one_usage else []

    if phase_one_result.not_found or not phase_one_result.synset_id:
        phase_one_result.raw_response = json.dumps(
            {"initial": phase_one_dict, "confirmation": None}
        )
        phase_one_result.usage_records = usage_records
        return phase_one_result

    synset_details = _lookup_wordnet_synset(phase_one_result.synset_id)
    details_text = _format_synset_details(synset_details, phase_one_result)

    confirmation_messages = [
        {
            "role": "system",
            "content": (
                "You are a meticulous linguist who double-checks WordNet mappings. "
                "Review the candidate synset and either confirm it or select a better match."
            ),
        },
        {
            "role": "user",
            "content": (
                "A previous pass selected the following synset for the product.\n"
                f"Product:\n{product.description}\n\n"
                f"Candidate synset: {phase_one_result.synset_id}\n"
                f"Model explanation: {phase_one_result.reason or 'n/a'}\n"
                f"Details:\n{details_text}\n\n"
                "If the synset is appropriate, return it unchanged using the record_synset function. "
                "Otherwise return a better matching synset."
            ),
        },
    ]

    phase_two_result, phase_two_dict, phase_two_usage = _invoke_synset_phase(
        client,
        messages=confirmation_messages,
        model=model,
        phase="confirmation",
    )
    if phase_two_usage:
        usage_records.append(phase_two_usage)

    phase_two_result.raw_response = json.dumps(
        {"initial": phase_one_dict, "confirmation": phase_two_dict}
    )
    phase_two_result.usage_records = usage_records
    return phase_two_result


def _format_synset_details(
    details: Optional[Dict[str, object]], candidate: SynsetResult
) -> str:
    if not details:
        fragments = []
        if candidate.synset_name:
            fragments.append(f"Name: {candidate.synset_name}")
        if candidate.synset_definition:
            fragments.append(f"Definition: {candidate.synset_definition}")
        if not fragments:
            return "Synset details unavailable."
        return "\n".join(fragments)

    pieces = [
        f"Name: {details.get('name') or candidate.synset_name or 'unknown'}",
        f"Definition: {details.get('definition') or candidate.synset_definition or 'n/a'}",
    ]
    lemmas = details.get("lemmas")
    if lemmas:
        pieces.append("Lemmas: " + ", ".join(sorted(str(lemma) for lemma in lemmas)))
    examples = details.get("examples")
    if examples:
        formatted = "\n".join(f"- {example}" for example in examples)
        pieces.append(f"Usage examples:\n{formatted}")
    return "\n".join(pieces)


def _lookup_wordnet_synset(synset_id: Optional[str]) -> Optional[Dict[str, object]]:
    if not synset_id or len(synset_id) < 2:
        return None
    pos, offset_text = synset_id[0], synset_id[1:]
    try:
        offset = int(offset_text)
    except ValueError:
        return None
    try:  # pragma: no cover - optional dependency
        from nltk.corpus import wordnet as wn  # type: ignore
    except (ImportError, LookupError):
        return None
    try:
        synset = wn.synset_from_pos_and_offset(pos, offset)
    except Exception:  # pragma: no cover - guard against unexpected formats
        return None
    return {
        "name": synset.name().split(".")[0],
        "definition": synset.definition(),
        "lemmas": [lemma.name().replace("_", " ") for lemma in synset.lemmas()],
        "examples": list(synset.examples()),
    }


def _invoke_synset_phase(
    client: "OpenAI",
    *,
    messages: list[Dict[str, object]],
    model: str,
    phase: str,
) -> tuple[SynsetResult, Dict[str, object], Optional[UsageRecord]]:
    tool_choice = {"type": "function", "name": "record_synset"}
    response = client.responses.create(
        model=model,
        input=messages,
        tools=[_build_tool_spec()],
        tool_choice=tool_choice,
    )

    response_dict = response.model_dump()
    tool_payload = _extract_tool_response(response_dict)
    result = _build_synset_result(tool_payload, response_dict)
    usage_record = _extract_usage(response_dict, phase=phase, model=model)
    return result, response_dict, usage_record


def _build_synset_result(tool_payload: Dict[str, object], response_dict: Dict[str, object]) -> SynsetResult:
    not_found = bool(tool_payload.get("not_found"))
    synset_id = tool_payload.get("synset_id")
    synset_name = tool_payload.get("synset_name")
    synset_definition = tool_payload.get("synset_definition")
    confidence = tool_payload.get("confidence")
    reason = tool_payload.get("reason" or None)

    if not not_found:
        if not synset_id:
            not_found = True
            missing_reason = (
                "Model response indicated a synset was found but did not include a synset_id."
            )
            reason = f"{reason}\n{missing_reason}" if reason else missing_reason
        elif confidence is not None and not isinstance(confidence, (int, float)):
            raise ValueError("Confidence value must be numeric when provided")

    if not_found:
        synset_id = None
        synset_name = None
        synset_definition = None
        confidence = None

    return SynsetResult(
        synset_id=synset_id,
        synset_name=synset_name,
        synset_definition=synset_definition,
        confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
        not_found=not_found,
        reason=reason,
        raw_response=json.dumps(response_dict),
    )


def _extract_usage(
    response_dict: Dict[str, object], *, phase: str, model: str
) -> Optional[UsageRecord]:
    usage = response_dict.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    return UsageRecord(
        phase=phase,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        recorded_at=datetime.now(timezone.utc),
    )


def _parse_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.fromisoformat(value.replace(" ", "T"))
        except ValueError:
            return None


def process_products(
    *,
    csv_path: Path,
    database_path: Path,
    batch_size: int,
    model: str,
    client: Optional["OpenAI"] = None,
) -> int:
    """Process up to ``batch_size`` unprocessed products."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if client is None:
        if OpenAI is None:  # pragma: no cover - requires optional dependency
            raise RuntimeError("The openai package is required to contact the API")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable must be set to contact the API")
        client = OpenAI()

    source = CSVProductSource(csv_path)
    database = SynsetDatabase(database_path)

    try:
        processed_ids = database.processed_product_ids()
        processed_count = 0

        for product in source.iter_products():
            if product.product_id in processed_ids:
                continue
            result = call_synset_model(client, product, model=model)
            database.store_result(product, result, model)
            if result.usage_records:
                database.store_usage(product.product_id, result.usage_records)
                _maybe_throttle(database)
            processed_ids.add(product.product_id)
            processed_count += 1
            if processed_count >= batch_size:
                break
    finally:
        database.close()

    return processed_count


def _maybe_throttle(database: SynsetDatabase) -> None:
    threshold_per_day = 5_000_000
    window_seconds = 24 * 3600
    stats = database.recent_usage(window_seconds)
    if not stats:
        return
    now = datetime.now(timezone.utc)
    elapsed = max((now - stats.earliest).total_seconds(), 1.0)
    tokens_per_second = stats.total_tokens / elapsed
    max_tokens_per_second = threshold_per_day / window_seconds
    if tokens_per_second <= max_tokens_per_second:
        return
    required_elapsed = stats.total_tokens / max_tokens_per_second
    sleep_seconds = math.ceil(required_elapsed - elapsed)
    if sleep_seconds <= 0:
        return
    time.sleep(sleep_seconds)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify products into WordNet synsets and store them in SQLite.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("products_point_one_percent_sample.csv"),
        help="Path to the products CSV file.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/product_synsets.sqlite"),
        help="Path to the SQLite database that will store the results.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1000,
        help="Number of unprocessed products to classify in this run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model to use for synset selection.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    processed = process_products(
        csv_path=args.csv,
        database_path=args.database,
        batch_size=args.batch,
        model=args.model,
    )
    print(f"Processed {processed} products into synsets.")


if __name__ == "__main__":
    main()
