"""Transformer for the MixtureVitae Web-Rewrite subset."""

from __future__ import annotations

from core.registry import BatchType, TransformResult, register_transformer

CHUNK_SIZE = 20_000
SEPARATOR = "\n\n<|endoftext|>\n\n"
TEXT_COLUMN = "text"

def _normalize_sample(value: object) -> str:
    """Convert one raw dataset value into a packable text sample."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)

def _pack_samples(samples: list[str], repo_id: str) -> TransformResult:
    """Pack batch-local samples into approximately 20k-character chunks."""
    texts: list[str] = []
    current_chunk = ""

    for sample in samples:
        if not sample:
            continue

        if not current_chunk:
            current_chunk = sample
            continue

        candidate = current_chunk + SEPARATOR + sample
        if len(candidate) > CHUNK_SIZE:
            texts.append(current_chunk)
            current_chunk = sample
        else:
            current_chunk = candidate

    if current_chunk:
        texts.append(current_chunk)

    return {"text": texts, "source": [repo_id] * len(texts)}

@register_transformer("mixture-vitae-backup/MixtureVitae-2TT/data/web-rewrite")
def transform_web_rewrite(batch: BatchType, repo_id: str) -> TransformResult:
    """Transform pre-flattened Web-Rewrite rows into packed 20k text chunks."""
    raw_texts = batch.get(TEXT_COLUMN)
    if raw_texts is None:
        return {"text": [], "source": []}

    flattened_samples = [_normalize_sample(value) for value in raw_texts]
    return _pack_samples(flattened_samples, repo_id)