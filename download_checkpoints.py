"""Download the archived baselines at one pinned Hub revision, without evaluating."""

import argparse
import hashlib
import json
import os
from pathlib import Path

# Plain HTTP avoids optional Xet/CAS authentication failures on public files.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "dsa4212-tinystories/tinystories-nanogpt"
DEFAULT_RUNS = ["large_data100", "xl_data50", "xl_data100", "xxl_data100"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", default="hf_cache")
    parser.add_argument("--revision", help="Default: reuse the manifest revision, or pin the current Hub commit")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    revision = args.revision or manifest.get("revision")
    info = HfApi().model_info(REPO_ID, revision=revision, files_metadata=True)
    if manifest and manifest["revision"] != info.sha:
        parser.error("Choose a new output directory for a different Hub revision")
    manifest.update(repo_id=REPO_ID, revision=info.sha)
    manifest.setdefault("files", {})
    available = {file.rfilename: file for file in info.siblings}
    print(f"Hub revision: {info.sha}", flush=True)
    for run in args.runs:
        for filename in ("checkpoint.pt", "run_config.json", "metrics.csv"):
            relative = f"{run}/{filename}"
            if relative not in available:
                parser.error(f"Missing file on Hub: {relative}")
            print(f"Downloading/verifying {relative} ({available[relative].size:,} bytes)", flush=True)
            path = hf_hub_download(REPO_ID, relative, revision=info.sha, local_dir=output)
            digest = sha256_file(path)
            remote_hash = (available[relative].lfs or {}).get("sha256")
            if remote_hash is not None and digest != remote_hash:
                raise RuntimeError(f"SHA-256 mismatch for {path}")
            manifest["files"][relative] = {"bytes": Path(path).stat().st_size, "sha256": digest}
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(manifest, indent=2) + "\n")
            temporary.replace(manifest_path)
    print(f"Saved checksums and revision to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
