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

for run in runs:
    local_dir = f"runs/followups_1h/{run}"
    repo_dir = f"followups_1h/{run}"

    print(f"Uploading {run}...", flush=True)

    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=repo_dir,
        commit_message=f"Upload follow-up checkpoint {run}",
    )

    print(f"Finished {run}", flush=True)
