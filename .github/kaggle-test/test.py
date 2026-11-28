import platform
import subprocess
import time

print("python", platform.python_version())
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "no GPU visible")
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception as e:
    print("torch import failed", e)
print("ok at", time.ctime())
