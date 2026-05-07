#!/usr/bin/env python3
"""Main orchestration pipeline for downloading, transforming, and persisting datasets.

This version adds:
1. Shard-based resume support for large dataset configs.
2. Atomic writes using temporary files before final promotion.

Each config is loaded once, then processed shard by shard. Existing shard files
are skipped unless --overwrite is enabled.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
from pathlib import Path
from typing import Optional

from datasets import (
    Dataset,
    DownloadConfig,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)

from core.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OUT_DIR,
    LOG_FILE_NAME,
    SOURCE_DATASETS,
    TARGET_COLUMNS,
    SourceSpec,
)
from core.registry import discover_transformers, get_transformer
from core.utils import retry_call, setup_logging, slugify


DEFAULT_SHARD_SIZE = 500_000


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download and transform all source dataset configs into local "
            "sharded .jsonl.gz files."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for .jsonl.gz files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Local datasets cache directory.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of worker processes for datasets.map().",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size passed to datasets.map(batched=True).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Maximum number of records per output shard.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for transient Hub operations.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=5.0,
        help="Base backoff in seconds for retryable operations.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        help="Optional Hugging Face token. Falls back to HF_TOKEN / HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional dataset revision (branch, tag, or commit SHA).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output shard files that already exist.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Giới hạn số lượng shard tối đa cần xử lý để không phải chạy hết",
    )
    return parser.parse_args()


def get_source_configs(spec: SourceSpec, args: argparse.Namespace) -> list[Optional[str]]:
    if spec.force_default_only or spec.data_dir is not None:
        return [None]

    configs = list(
        retry_call(
            lambda: get_dataset_config_names(
                spec.repo_id,
                revision=args.revision,
                token=args.token,
            ),
            max_retries=args.max_retries,
            base_delay_seconds=args.retry_backoff_seconds,
            operation_name=f"get_dataset_config_names({spec.repo_id})",
        )
    )

    if spec.include_configs is not None:
        include_set = set(spec.include_configs)
        missing_configs = include_set.difference(configs)
        if missing_configs:
            raise ValueError(
                f"Configured include_configs not found for {spec.repo_id}: "
                f"{sorted(missing_configs)}. Available configs: {configs}"
            )
        configs = [config for config in configs if config in include_set]

    if spec.exclude_configs:
        exclude_set = set(spec.exclude_configs)
        configs = [config for config in configs if config not in exclude_set]

    if spec.skip_first_configs:
        configs = configs[spec.skip_first_configs:]

    return configs


def choose_split(repo_id: str, config_name: Optional[str], args: argparse.Namespace) -> str:
    """Choose the most appropriate split for a dataset config."""
    split_names = retry_call(
        lambda: get_dataset_split_names(
            repo_id,
            config_name=config_name,
            revision=args.revision,
            token=args.token,
        ),
        max_retries=args.max_retries,
        base_delay_seconds=args.retry_backoff_seconds,
        operation_name=f"get_dataset_split_names({repo_id}, {config_name or 'default'})",
    )
    available_splits = list(split_names)

    for preferred_split in ("train", "validation", "test"):
        if preferred_split in available_splits:
            return preferred_split

    if not available_splits:
        raise ValueError(f"No splits found for {repo_id} / {config_name or 'default'}")

    return available_splits[0]


def build_shard_output_path(
    out_dir: Path,
    repo_id: str,
    config_name: Optional[str],
    data_dir: Optional[str],
    shard_index: int,
) -> Path:
    """Build the final output path for one processed shard."""
    repo_basename = slugify(repo_id.split("/", 1)[1])
    config_label = slugify(config_name or "default")
    
    # Nếu có data_dir (như data/tools hay data/web-rewrite), slugify và chèn vào tên
    if data_dir:
        dir_label = slugify(data_dir)
        filename = f"{repo_basename}_{dir_label}_{config_label}_shard_{shard_index:05d}.jsonl.gz"
    else:
        filename = f"{repo_basename}_{config_label}_shard_{shard_index:05d}.jsonl.gz"
        
    return out_dir / filename


def build_temporary_output_path(output_path: Path) -> Path:
    """Build the temporary output path used for atomic writes."""
    return output_path.with_name(f"{output_path.name}.tmp")


def select_contiguous_shard(dataset: Dataset, shard_index: int, num_shards: int) -> Dataset:
    """Select one contiguous shard from a dataset."""
    return dataset.shard(
        num_shards=num_shards,
        index=shard_index,
        contiguous=True,
    )


def write_shard_atomically(
    shard_dataset: Dataset,
    output_path: Path,
    *,
    repo_id: str,
    config_label: str,
    shard_index: int,
    args: argparse.Namespace,
    data_dir: str | None = None,
) -> None:
    """Transform one shard and write it atomically to the final output path.

    The write flow is:
    1. Transform the shard with datasets.map().
    2. Write to a temporary file ending with .tmp.
    3. Promote the temporary file to the final target with os.replace().
    """
    registry_key = f"{repo_id}/{data_dir}" if data_dir else repo_id
    transformer = get_transformer(registry_key)
    temporary_output_path = build_temporary_output_path(output_path)

    if temporary_output_path.exists():
        logging.warning(
            "Found stale temporary shard file. Replacing it: %s",
            temporary_output_path,
        )
        temporary_output_path.unlink()

    transformed = shard_dataset.map(
        lambda batch: transformer(batch, repo_id),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=shard_dataset.column_names,
        desc=f"Transforming {repo_id} / {config_label} / shard {shard_index}",
    )

    if list(transformed.column_names) != TARGET_COLUMNS:
        transformed = transformed.select_columns(TARGET_COLUMNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        transformed.to_json(
            str(temporary_output_path),
            compression="gzip",
            force_ascii=False,
        )
        os.replace(str(temporary_output_path), str(output_path))
    except Exception:
        logging.exception(
            "Atomic write failed for repo=%s config=%s shard=%s",
            repo_id,
            config_label,
            shard_index,
        )
        raise
    finally:
        del transformed
        gc.collect()

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logging.info(
        "Saved repo=%s config=%s shard=%s rows=%s size=%.2f MiB -> %s",
        repo_id,
        config_label,
        shard_index,
        shard_dataset.num_rows,
        file_size_mb,
        output_path,
    )


def process_one_config(spec: SourceSpec, config_name: Optional[str], args: argparse.Namespace) -> None:
    repo_id = spec.repo_id

    if args.shard_size <= 0:
        raise ValueError("--shard-size must be greater than 0")

    config_label = config_name or "default"
    split_name = "train" if spec.data_dir is not None else choose_split(repo_id, config_name, args)

    logging.info(
        "Loading repo=%s config=%s split=%s data_dir=%s",
        repo_id,
        config_label,
        split_name,
        spec.data_dir,
    )

    download_config = DownloadConfig(
        resume_download=True,
        max_retries=max(1, args.max_retries),
    )

    dataset = retry_call(
        lambda: load_dataset(
            repo_id,
            name=config_name,
            data_dir=spec.data_dir,
            split=split_name,
            cache_dir=str(args.cache_dir),
            token=args.token,
            revision=args.revision,
            download_config=download_config,
        ),
        max_retries=args.max_retries,
        base_delay_seconds=args.retry_backoff_seconds,
        operation_name=f"load_dataset({repo_id}, {config_label}, split={split_name}, data_dir={spec.data_dir})",
    )

    total_rows = dataset.num_rows
    if total_rows == 0:
        logging.warning(
            "Dataset is empty. Skipping repo=%s config=%s split=%s",
            repo_id,
            config_label,
            split_name,
        )
        del dataset
        gc.collect()
        return

    num_shards = math.ceil(total_rows / args.shard_size)
    actual_shards = num_shards
    if args.max_shards is not None:
        actual_shards = min(num_shards, args.max_shards)

    logging.info(
        "Processing repo=%s config=%s split=%s total_rows=%s shard_size=%s num_shards=%s",
        repo_id,
        config_label,
        split_name,
        total_rows,
        args.shard_size,
        actual_shards,
    )

    try:
        for shard_index in range(actual_shards):
            output_path = build_shard_output_path(
                args.out_dir,
                repo_id,
                config_name,
                spec.data_dir,
                shard_index,
            )

            if output_path.exists() and not args.overwrite:
                logging.info(
                    "Skipping shard %s for repo=%s config=%s because %s already exists.",
                    shard_index,
                    repo_id,
                    config_label,
                    output_path,
                )
                continue

            shard_dataset = select_contiguous_shard(
                dataset=dataset,
                shard_index=shard_index,
                num_shards=num_shards,
            )

            logging.info(
                "Processing repo=%s config=%s shard=%s/%s shard_rows=%s -> %s",
                repo_id,
                config_label,
                shard_index + 1,
                num_shards,
                shard_dataset.num_rows,
                output_path,
            )

            try:
                write_shard_atomically(
                    shard_dataset=shard_dataset,
                    output_path=output_path,
                    repo_id=repo_id,
                    config_label=config_label,
                    shard_index=shard_index,
                    args=args,
                    data_dir=spec.data_dir,
                )
            finally:
                del shard_dataset
                gc.collect()
    finally:
        del dataset
        gc.collect()


def main() -> int:
    """Entry point for the dataset conversion pipeline."""
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.out_dir / LOG_FILE_NAME)

    discover_transformers("dataset_transformers")

    logging.info("Phase 1 started.")
    logging.info("Output directory: %s", args.out_dir.resolve())
    logging.info("Cache directory: %s", args.cache_dir.resolve())
    logging.info(
        "num_proc=%s batch_size=%s shard_size=%s overwrite=%s",
        args.num_proc,
        args.batch_size,
        args.shard_size,
        args.overwrite,
    )

    total_configs = 0
    failed_configs = 0

    for spec in SOURCE_DATASETS:
        try:
            configs = get_source_configs(spec, args)
        except KeyboardInterrupt:
            logging.warning("Interrupted by user. Exiting.")
            raise
        except Exception:
            failed_configs += 1
            logging.exception(
                "Failed to enumerate configs for %s. Continuing with next dataset.",
                spec.repo_id,
            )
            continue

        logging.info("Discovered %s configs for %s", len(configs), spec.repo_id)

        for config_name in configs:
            total_configs += 1
            try:
                process_one_config(spec, config_name, args)
            except KeyboardInterrupt:
                logging.warning("Interrupted by user. Exiting.")
                raise
            except Exception:
                failed_configs += 1
                logging.exception(
                    "Failed while processing repo=%s config=%s. Continuing with next config.",
                    spec.repo_id,
                    config_name or "default",
                )

    succeeded = total_configs - failed_configs
    logging.info(
        "Phase 1 completed. total_configs=%s succeeded=%s failed=%s",
        total_configs,
        succeeded,
        failed_configs,
    )

    return 0 if failed_configs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())