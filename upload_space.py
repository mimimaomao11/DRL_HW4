"""
Upload hf_space/ to Hugging Face Spaces as mimimaomao11/drl-hw4-demo
Run after: python -c "from huggingface_hub import login; login(token='hf_xxxx')"
"""
from huggingface_hub import HfApi, create_repo
import os

REPO_ID  = "mimimaomao11/drl-hw4-demo"
SPACE_DIR = os.path.join(os.path.dirname(__file__), "hf_space")

api = HfApi()

# 1. Create Space (skip if already exists)
print(f"Creating Space: {REPO_ID} ...")
try:
    create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=False,
    )
    print("  Space created (or already exists)")
except Exception as e:
    print(f"  Note: {e}")

# 2. Upload all files
print(f"\nUploading files from {SPACE_DIR} ...")
for root, dirs, files in os.walk(SPACE_DIR):
    for fname in files:
        local_path = os.path.join(root, fname)
        # Relative path inside the Space repo
        rel_path = os.path.relpath(local_path, SPACE_DIR).replace("\\", "/")
        print(f"  Uploading: {rel_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=rel_path,
            repo_id=REPO_ID,
            repo_type="space",
        )

print(f"""
Done!
Space URL : https://huggingface.co/spaces/{REPO_ID}
Direct app: https://{REPO_ID.replace('/', '-')}.hf.space

Wait 2-3 minutes for the Space to build, then open the URL above.
""")
