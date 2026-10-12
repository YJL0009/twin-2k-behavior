#!/usr/bin/env python3
"""Collect aggregate results without participant responses or model weights."""
import argparse
from pathlib import Path
import zipfile


def collect(folder):
    folder = Path(folder).resolve()
    if not (folder / "complete.json").is_file():
        raise ValueError("Comparisons are not complete")
    destination = folder / "comparison_results.zip"
    files = [p for p in folder.iterdir() if p.is_file() and p.name in {
        "comparison.csv", "prediction_agreement.csv", "comparison_coverage.csv", "report.txt",
        "comparison_config.json", "complete.json", "retrieval_and_price_audit.json",
        "original_prompt_verification.json", "stage_metadata.json"}]
    for scores in sorted(folder.glob("*/scores")):
        files.extend(p for p in scores.iterdir() if p.is_file() and p.name in {
            "summary.json", "metrics_by_family.csv", "metrics_by_question.csv",
            "metrics_by_item.csv", "human_reference_by_question.csv", "matrix_exact.csv"})
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(folder)))
    temporary.replace(destination)
    return destination


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    print(collect(p.parse_args().folder))
