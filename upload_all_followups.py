from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()
repo_id = "dsa4212-tinystories/tinystories-nanogpt"

runs = [
    "xl_full_1h_seed42",
    "xl_full_1h_seed43",
    "xl_full_1h_seed44",
    "xxl_full_1h_seed42",
    "xxl_full_1h_seed43",
    "xxl_full_1h_seed44",
]

# Verify all required files before uploading
for run in runs:
    checkpoint = Path(f"runs/followups_1h/{run}/checkpoint.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

for folder in [
    "analysis/followups/seed_validation",
    "analysis/followups/archived_validation",
]:
    path = Path(folder)
    if not path.is_dir():
        raise FileNotFoundError(path)
    for name in ["protocol.json", "paired_comparisons.json"]:
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)

# Upload six model runs
for run in runs:
    print(f"Uploading checkpoint: {run}", flush=True)
    api.upload_folder(
        folder_path=f"runs/followups_1h/{run}",
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=f"followups_1h/{run}",
        commit_message=f"Upload {run}",
    )

# Upload evaluation files
for name in ["seed_validation", "archived_validation"]:
    print(f"Uploading {name}", flush=True)
    api.upload_folder(
        folder_path=f"analysis/followups/{name}",
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=f"analysis/followups/{name}",
        commit_message=f"Upload {name} evaluation files",
    )

# Upload summary
summary = Path("analysis/followups/seed_summary.md")
if summary.is_file():
    api.upload_file(
        path_or_fileobj=str(summary),
        path_in_repo="analysis/followups/seed_summary.md",
        repo_id=repo_id,
        repo_type="model",
    )

print("All uploads completed!", flush=True)
