"""Configuration objects and constants for the dataset conversion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


TARGET_COLUMNS: list[str] = ["text", "source"]
DEFAULT_OUT_DIR: Path = Path("./processed_datasets")
DEFAULT_CACHE_DIR: Path = Path("./hf_cache")
LOG_FILE_NAME: str = "phase1_convert.log"


@dataclass(frozen=True)
class SourceSpec:
    """Specification for a source dataset repository."""

    repo_id: str
    force_default_only: bool = False
    data_dir: str | None = None
    include_configs: tuple[str, ...] | None = None
    exclude_configs: tuple[str, ...] = ()
    skip_first_configs: int = 0


SOURCE_DATASETS: tuple[SourceSpec, ...] = (
    SourceSpec("ontocord/MixtureVitae-v1", force_default_only=True, data_dir="data/software"),
    SourceSpec("ontocord/MixtureVitae-v1", force_default_only=True, data_dir="data/math"),
    SourceSpec("ontocord/MixtureVitae-v1", force_default_only=True, data_dir="data/synthetic_instruct"),
    SourceSpec("laion/Scientific-Summaries"),
)