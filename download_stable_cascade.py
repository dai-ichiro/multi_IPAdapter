import os
from huggingface_hub import snapshot_download

repo_id="h94/IP-Adapter-FaceID"
folder = os.path.basename(repo_id)

snapshot_download(
    repo_id=repo_id,
    allow_patterns=[
        "*sdxl*"
    ],
    local_dir=folder,
    local_dir_use_symlinks=False
)

