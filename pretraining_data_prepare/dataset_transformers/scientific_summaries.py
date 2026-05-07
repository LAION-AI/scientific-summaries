"""Transformer for the laion/Scientific-Summaries dataset.

This module is designed for batched execution with datasets.map(batched=True).
It extracts only a strict allowlist of summary-oriented fields, explicitly
excluding author-related columns to avoid introducing scientist-name bias into
LLM training data.
"""

from __future__ import annotations

import html
import json
import random
from collections import defaultdict
from typing import TypeAlias

from core.registry import register_transformer


BatchType: TypeAlias = dict[str, list[object]]
TransformResult: TypeAlias = dict[str, list[str]]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

UNKNOWN_GROUP = "Unknown"

TARGET_FIELDS: tuple[str, ...] = (
    "summary_title",
    "field_subfield",
    "type_of_paper",
    "executive_summary",
    "research_context",
    "research_question_hypothesis",
    "methodological_details",
    "procedures_architectures",
    "key_results",
    "interpretation_implications",
    "contradictions_limitations",
    "claims",
    "data_code_availability",
    "robustness_ablation_notes",
    "ethical_considerations",
    "key_figures_tables",
    "three_takeaways",
    "oa_is_retracted",
)

JSON_THRESHOLD = 0.80
YAML_THRESHOLD = 0.90


def _infer_batch_size(batch: BatchType) -> int:
    """Infer batch size from the first available column."""
    for column in batch.values():
        return len(column)
    return 0


def _normalize_group_key(value: object) -> str:
    """Normalize field_subfield into a stable grouping key."""
    if value is None:
        return UNKNOWN_GROUP

    if isinstance(value, str):
        normalized = value.strip()
        return normalized if normalized else UNKNOWN_GROUP

    normalized = str(value).strip()
    return normalized if normalized else UNKNOWN_GROUP


def _sanitize_value(value: object) -> JsonValue:
    """Recursively sanitize a raw dataset value into a JSON-safe structure.

    Empty strings, empty containers, and None are normalized to None so they can
    be filtered out before serialization.
    """
    if value is None:
        return None

    if isinstance(value, str):
        normalized = value.strip()
        return normalized if normalized else None

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value != value:
            return None
        return value

    if isinstance(value, dict):
        sanitized_dict: dict[str, JsonValue] = {}
        for key, item in value.items():
            sanitized_item = _sanitize_value(item)
            if sanitized_item is not None:
                sanitized_dict[str(key)] = sanitized_item
        return sanitized_dict or None

    if isinstance(value, (list, tuple)):
        sanitized_list: list[JsonValue] = []
        for item in value:
            sanitized_item = _sanitize_value(item)
            if sanitized_item is not None:
                sanitized_list.append(sanitized_item)
        return sanitized_list or None

    normalized = str(value).strip()
    return normalized if normalized else None


def _build_ordered_record(
    batch: BatchType,
    row_index: int,
    ordered_fields: list[str],
) -> dict[str, JsonValue]:
    """Build one ordered record using only allowlisted fields."""
    record: dict[str, JsonValue] = {}

    for field_name in ordered_fields:
        column = batch.get(field_name)
        if column is None or row_index >= len(column):
            continue

        sanitized_value = _sanitize_value(column[row_index])
        if sanitized_value is not None:
            record[field_name] = sanitized_value

    return record


def _yaml_quote_string(value: str) -> str:
    """Quote a YAML string safely using JSON escaping rules."""
    return json.dumps(value, ensure_ascii=False)


def _yaml_scalar(value: JsonScalar) -> str:
    """Serialize a scalar value into YAML-safe text."""
    if value is None:
        return "null"

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        return repr(value)

    return _yaml_quote_string(value)


def _yaml_lines(value: JsonValue, indent: int = 0) -> list[str]:
    """Recursively serialize a JSON-like value into YAML lines."""
    prefix = " " * indent

    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]

        lines: list[str] = []
        for key, item in value.items():
            escaped_key = str(key)
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{escaped_key}:")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{escaped_key}: {_yaml_scalar(item)}")
        return lines

    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]

        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines

    return [f"{prefix}{_yaml_scalar(value)}"]


def _serialize_json(record: dict[str, JsonValue]) -> str:
    """Serialize a record as compact JSON."""
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def _serialize_yaml(record: dict[str, JsonValue]) -> str:
    """Serialize a record as simple YAML without external dependencies."""
    return "\n".join(_yaml_lines(record))


def _serialize_html(record: dict[str, JsonValue]) -> str:
    """Serialize a record as basic HTML."""
    parts = ["<article>"]

    for key, value in record.items():
        if isinstance(value, (dict, list)):
            rendered_value = json.dumps(value, ensure_ascii=False)
        elif value is None:
            rendered_value = ""
        else:
            rendered_value = str(value)

        parts.append(
            f"<div><b>{html.escape(key)}</b>: "
            f"{html.escape(rendered_value)}</div>"
        )

    parts.append("</article>")
    return "".join(parts)


def _serialize_record(
    record: dict[str, JsonValue],
    rng: random.Random,
) -> str:
    """Serialize a record into JSON, YAML, or HTML with weighted randomization."""
    format_selector = rng.random()

    if format_selector < JSON_THRESHOLD:
        return _serialize_json(record)

    if format_selector < YAML_THRESHOLD:
        return _serialize_yaml(record)

    return _serialize_html(record)


@register_transformer("laion/Scientific-Summaries")
def transform_scientific_summaries(batch: BatchType, repo_id: str) -> TransformResult:
    """Transform Scientific-Summaries rows into shuffled structured text outputs.

    Processing steps:
    1. Group rows by field_subfield, defaulting missing values to "Unknown".
    2. Generate one shuffled field order per group and reuse it for every row
       in that group.
    3. Serialize each filtered record as JSON (~80%), YAML (~10%), or HTML (~10%).
    4. Return the standardized {"text": [...], "source": [...]} structure.

    Only TARGET_FIELDS are retained. Author-related fields are intentionally
    excluded and never accessed.
    """
    batch_size = _infer_batch_size(batch)
    if batch_size == 0:
        return {"text": [], "source": []}

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    field_subfield_column = batch.get("field_subfield", [None] * batch_size)

    for row_index in range(batch_size):
        group_key = _normalize_group_key(field_subfield_column[row_index])
        grouped_indices[group_key].append(row_index)

    texts: list[str] = [""] * batch_size
    sources: list[str] = [repo_id] * batch_size
    rng = random.Random()

    for group_key, row_indices in grouped_indices.items():
        ordered_fields = list(TARGET_FIELDS)
        rng.shuffle(ordered_fields)

        for row_index in row_indices:
            ordered_record = _build_ordered_record(
                batch=batch,
                row_index=row_index,
                ordered_fields=ordered_fields,
            )
            texts[row_index] = _serialize_record(ordered_record, rng)

    return {"text": texts, "source": sources}