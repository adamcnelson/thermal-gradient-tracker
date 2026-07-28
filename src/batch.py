"""
Batch processing utilities: find .seq files and process them in sequence.
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .arena_mask import ArenaMask, TrackingConfig
from .tracking import track_file

log = logging.getLogger("thermal-tracker")


def find_seq_files(root: str, recursive: bool = False) -> List[Path]:
    """Return all .seq files under root, sorted by path."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    pattern = "**/*.seq" if recursive else "*.seq"
    files = sorted(root_path.glob(pattern))
    if not files:
        log.warning(f"No .seq files found under {root}")
    return files


def output_csv_path(seq_path: Path, output_dir: Path, interval: int) -> Path:
    """Compute the expected output CSV path for a given .seq file."""
    stem = seq_path.stem
    name = f"{stem}_tracking_every{interval}frames.csv"
    return output_dir / name


def process_batch(
    seq_files: List[Path],
    config: TrackingConfig,
    output_dir: str,
    overwrite: bool = False,
    save_qc_images: bool = True,
    n_qc_images: int = 20,
    save_qc_summary: bool = True,
    save_plots: bool = True,
) -> List[dict]:
    """
    Process a list of .seq files.

    Returns a list of per-file result dicts with keys:
      seq_path, csv_path, status, error
    """
    from .qc_outputs import save_qc_summary as write_qc_summary, save_diagnostic_plots

    out_dir = Path(output_dir)
    results = []

    for i, seq_path in enumerate(seq_files, 1):
        log.info(f"\n[{i}/{len(seq_files)}] {seq_path.name}")

        result = {
            "seq_path": str(seq_path),
            "csv_path": None,
            "status": "pending",
            "error": None,
        }

        # Check if output already exists
        csv_path = output_csv_path(seq_path, out_dir, config.sampling_interval_frames)
        if csv_path.exists() and not overwrite:
            log.info(f"  Skipping (output exists): {csv_path.name}")
            result["status"] = "skipped"
            result["csv_path"] = str(csv_path)
            results.append(result)
            continue

        try:
            from .seq_io import SeqReader

            with SeqReader(str(seq_path)) as reader:
                frame_shape = reader.frame_shape

            arena_mask = ArenaMask.from_config(config, frame_shape)

            df, csv_out, entry_info = track_file(
                seq_path=str(seq_path),
                config=config,
                arena_mask=arena_mask,
                output_dir=str(out_dir),
                overwrite=overwrite,
                save_qc_images=save_qc_images,
                n_qc_images=n_qc_images,
            )

            if save_qc_summary:
                summary_dir = out_dir / "qc_summaries"
                summary_dir.mkdir(parents=True, exist_ok=True)
                write_qc_summary(
                    df,
                    str(summary_dir / f"{seq_path.stem}_qc_summary.csv"),
                    entry_info=entry_info,
                )

            if save_plots:
                plot_dir = out_dir / "qc_plots"
                plot_dir.mkdir(parents=True, exist_ok=True)
                save_diagnostic_plots(df, str(plot_dir / seq_path.stem))

            result["status"] = "ok"
            result["csv_path"] = str(csv_out)

        except FileExistsError as e:
            log.warning(f"  Skipped: {e}")
            result["status"] = "skipped"
            result["error"] = str(e)

        except Exception as e:
            log.error(f"  ERROR processing {seq_path.name}: {e}", exc_info=True)
            result["status"] = "error"
            result["error"] = str(e)

        results.append(result)

    # Print final summary
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    log.info(f"\nBatch complete: {ok} processed, {skipped} skipped, {errors} errors")
    return results
