#!/usr/bin/env python3
"""Download the pinned wave-split data needed by the experiment source package.

No participant data are included in the public source archive. Hugging Face is
only used when this command runs; existing local files can instead be selected
with ``prepare_data.py --dataset-root /path/to/Twin-2K-500``.
"""
import argparse
import hashlib
import json
from pathlib import Path

REPO_ID = "LLM-Digital-Twin/Twin-2K-500"
REVISION = "f883165a3026fde855dfd448e0cd16443ab257b6"
FILES = {
    "question_catalog_and_human_response_csv/question_catalog.json": "e72e305f0c7c83bb990e97de4c0cf2c718ead18cc197239b7feeb354c9ac1bef",
    "question_catalog_and_human_response_csv/wave1_3_response.csv": "fc227b1682e2654fd4b2a8152294b68118c0e3be4d83f4a9af5f3e96df56d3f0",
    "question_catalog_and_human_response_csv/wave4_response.csv": "09ec1a88bdd48fec95963da3102df0ca467378bf729cbc1c0b869b0f1e4fb942",
    "wave_split/chunks/wave_persona_chunk_001.parquet": "dd92be4ac50d3476dc0166f7b6b2fdb07ff6e167c379f62d1b7730905135753b",
    "wave_split/chunks/wave_persona_chunk_002.parquet": "77ed5561e0f02e07237437b1ce88a409f02a02839e7400ad392bc24193469bf1",
    "wave_split/chunks/wave_persona_chunk_003.parquet": "c856865424fea528c928c4dd5e29efb0c69a8fb7bf79d5d17443e321c1083bd9",
    "wave_split/chunks/wave_persona_chunk_004.parquet": "3c88d726d7b5f8bda1e950c5ef00e757bef2abebfbdce144f9c81d7c4ff36f33",
    "wave_split/chunks/wave_persona_chunk_005.parquet": "2b8d46f79c9e25e58aae266397e46d572bda27f42c4b18fadc8f110f3ad833d7",
    "wave_split/chunks/wave_persona_chunk_006.parquet": "f7e7a94a900596d7682f389ccab5c3bee3a85737d9fb08828937ce4c45c90abe",
    "wave_split/chunks/wave_persona_chunk_007.parquet": "919910a97d84fc8b437d380dba8aebea3430436e382419a9874a0485109bf296",
}


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_dataset(folder):
    """Verify immutable raw files; reading a file hash does not analyze answers."""
    folder = Path(folder)
    observed = {}
    for relative, expected in FILES.items():
        path = folder / relative
        if not path.is_file():
            raise ValueError(f"Missing dataset file: {path}")
        actual = digest_file(path)
        if actual != expected:
            raise ValueError(f"Pinned dataset SHA256 mismatch: {relative}")
        observed[relative] = actual
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--destination", type=Path,
                        help="Default: ROOT/data/Twin-2K-500")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    target = (args.destination or args.root / "data/Twin-2K-500").resolve()
    if not args.verify_only:
        from huggingface_hub import hf_hub_download
        target.mkdir(parents=True, exist_ok=True)
        for relative, expected in FILES.items():
            local = target / relative
            if local.is_file() and digest_file(local) == expected:
                print(f"Verified cached: {relative}", flush=True)
                continue
            hf_hub_download(repo_id=REPO_ID, repo_type="dataset", revision=REVISION,
                            filename=relative, local_dir=str(target))
    hashes = verify_dataset(target)
    manifest = {"repo_id": REPO_ID, "revision": REVISION,
                "purpose": "wave_split_training_and_label_tables",
                "full_persona_downloaded": False, "sha256": hashes}
    if not args.verify_only:
        (target / "source_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(hashes)} pinned files: {target}")
    print("Next: python prepare_data.py --root <experiment-directory>")


if __name__ == "__main__":
    main()
