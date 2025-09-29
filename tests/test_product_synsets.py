from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from padjective.product_synsets import (
    CSVProductSource,
    _extract_tool_response,
    process_products,
)


def test_csv_product_source_reads_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "products.csv"
    csv_path.write_text("title,tags\nWidget,tag1\nGadget,tag2\n", encoding="utf-8")

    source = CSVProductSource(csv_path)
    records = list(source.iter_products())

    assert [record.title for record in records] == ["Widget", "Gadget"]
    assert records[0].product_id == 1
    assert records[1].product_id == 2


def test_extract_tool_response_handles_nested_structure() -> None:
    payload = {
        "output": [
            {
                "content": [
                    {
                        "type": "tool_call",
                        "function": {
                            "name": "record_synset",
                            "arguments": json.dumps(
                                {
                                    "synset_id": "n00000000",
                                    "synset_name": "test",
                                    "synset_definition": "definition",
                                    "confidence": 0.5,
                                    "not_found": False,
                                }
                            ),
                        },
                    }
                ]
            }
        ]
    }

    result = _extract_tool_response(payload)
    assert result["synset_id"] == "n00000000"
    assert result["not_found"] is False


def test_extract_tool_response_handles_tool_use_content() -> None:
    payload = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "record_synset",
                        "input": {
                            "synset_id": "n04256520",
                            "synset_name": "scooter",
                            "synset_definition": "a small motorcycle with a step-through frame",
                            "confidence": 0.8,
                            "not_found": False,
                        },
                    }
                ],
            }
        ]
    }

    result = _extract_tool_response(payload)
    assert result["synset_name"] == "scooter"
    assert result["confidence"] == 0.8


def test_extract_tool_response_handles_function_call_items() -> None:
    payload = {
        "output": [
            {
                "type": "function_call",
                "name": "record_synset",
                "arguments": json.dumps(
                    {
                        "synset_id": "n01234567",
                        "synset_name": "gizmo",
                        "synset_definition": "a made-up gadget",
                        "confidence": 0.3,
                        "not_found": False,
                    }
                ),
            }
        ]
    }

    result = _extract_tool_response(payload)
    assert result["synset_definition"] == "a made-up gadget"
    assert result["confidence"] == 0.3


class MockClient:
    def __init__(self, response_payload: dict) -> None:
        self.response_payload = response_payload
        self.calls = []

    class Responses:
        def __init__(self, outer: "MockClient") -> None:
            self.outer = outer

        def create(self, **kwargs):
            self.outer.calls.append(kwargs)
            return MockClient._Response(self.outer.response_payload)

    class _Response:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def model_dump(self) -> dict:
            return self.payload

    @property
    def responses(self) -> "MockClient.Responses":
        return MockClient.Responses(self)


def test_process_products_stores_results(tmp_path: Path) -> None:
    csv_path = tmp_path / "products.csv"
    csv_path.write_text("title,tags\nWidget,tag1\n", encoding="utf-8")
    db_path = tmp_path / "results.sqlite"

    response_payload = {
        "output": [
            {
                "type": "tool_call",
                "function": {
                    "name": "record_synset",
                    "arguments": json.dumps(
                        {
                            "synset_id": "n00001740",
                            "synset_name": "entity",
                            "synset_definition": "that which is perceived or known or inferred to have its own distinct existence",
                            "confidence": 0.1,
                            "not_found": False,
                        }
                    ),
                },
            }
        ]
    }
    client = MockClient(response_payload)

    processed = process_products(
        csv_path=csv_path,
        database_path=db_path,
        batch_size=1,
        model="gpt-5-mini",
        client=client,
    )

    assert processed == 1

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT product_id, synset_id, not_found FROM product_synsets"
        )
        row = cursor.fetchone()
        assert row[0] == 1
        assert row[1] == "n00001740"
        assert row[2] == 0


@pytest.mark.parametrize("invalid_batch", [0, -5])
def test_process_products_validates_batch_size(tmp_path: Path, invalid_batch: int) -> None:
    csv_path = tmp_path / "products.csv"
    csv_path.write_text("title,tags\nWidget,tag1\n", encoding="utf-8")

    with pytest.raises(ValueError):
        process_products(
            csv_path=csv_path,
            database_path=tmp_path / "db.sqlite",
            batch_size=invalid_batch,
            model="gpt-5-mini",
            client=MockClient({"output": []}),
        )
