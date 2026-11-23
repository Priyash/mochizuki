"""The Modal app that runs one Mochizuki batch. Deployed by the produce workflow:

    modal deploy worker/modal_app.py

The batch's inputs (manifest.json, worker.py and its files) are already on the `mochizuki`
volume under /batches/<id>/dataset; results go to /batches/<id>/out. The same worker.py as
on Kaggle does the work, so both backends make identical outputs.
"""
import os
import shutil
import subprocess
import sys

import modal

PACKAGES = ["torch>=2.5", "diffusers>=0.36", "transformers>=4.51", "accelerate>=1.2", "gguf>=0.10",
            "bitsandbytes>=0.45", "spandrel>=0.4", "ftfy", "imageio-ffmpeg", "huggingface_hub>=0.30",
            "sentencepiece", "numpy", "pillow"]

image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")
         .pip_install(*PACKAGES)
         .env({"HF_HOME": "/tmp/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
volume = modal.Volume.from_name("mochizuki", create_if_missing=True)
app = modal.App("mochizuki-worker")


@app.function(image=image, gpu=os.environ.get("MOCHIZUKI_MODAL_GPU", "L4"), cpu=4.0, memory=49152,
              ephemeral_disk=200 * 1024, volumes={"/data": volume}, timeout=4 * 3600, retries=0)
def run_batch(batch_id: str, engine_set: str = "gpu") -> dict:
    volume.reload()
    base = f"/data/batches/{batch_id}"
    out = "/tmp/out"                      # written locally, copied to the volume once at the end
    code = subprocess.run([sys.executable, f"{base}/dataset/worker.py",
                           "--manifest", f"{base}/dataset/manifest.json", "--inputs", f"{base}/dataset",
                           "--out", out, "--engine-set", engine_set, "--expect-batch", batch_id]).returncode
    if os.path.isdir(out):
        shutil.copytree(out, f"{base}/out", dirs_exist_ok=True)
    volume.commit()
    return {"batch": batch_id, "exit": code}
