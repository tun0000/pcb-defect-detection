"""Deploy the Gradio Space to Hugging Face (completes plan.md SS 2.6's acceptance
criteria: local test -> deploy -> live URL works).

Uploads app/ (app.py, requirements.txt, README.md, examples/) as-is. Requires
HF_TOKEN in .env (write-scoped token) and scripts/upload_hf.py to have already
run at least once (the Space's app.py downloads best.onnx from that model repo).

    uv run --group hf python scripts/deploy_space.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"
SPACE_ID = "betty0/pcb-defect-detection"


def main() -> int:
    import os

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found - set it in .env first (see plan.md SS 2.7)")
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"creating/confirming repo {SPACE_ID} (space, public, sdk=gradio)...")
    repo_url = api.create_repo(
        repo_id=SPACE_ID, repo_type="space", space_sdk="gradio", private=False, exist_ok=True
    )
    print(f"repo: {repo_url}")

    print(f"uploading {APP_DIR} -> {SPACE_ID} ...")
    commit_info = api.upload_folder(
        folder_path=str(APP_DIR),
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="Deploy Gradio Space (Phase 2 step 2.6)",
    )
    print(f"done: {commit_info}")
    print(f"\nSpace page: https://huggingface.co/spaces/{SPACE_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
