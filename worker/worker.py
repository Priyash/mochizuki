"""Runs ON Kaggle (or locally with --engine-set stub). One session, one batch.

Reads manifest.json, makes every job it can before the deadline, and writes
results.json after each job, so a session that is cut off still reports what it made.

    python worker.py --manifest manifest.json --inputs DIR --out DIR [--engine-set stub]

Engines
  keyframe / variant : Qwen-Image-Edit-2511 (Apache 2.0), GGUF transformer if available
  take               : Wan 2.2 TI2V-5B or I2V-A14B (Apache 2.0), image-to-video
  upscale            : Real-CUGAN 3x via spandrel, then exact 3840x2160 (cover + centre crop)

This file has not been run on a GPU by its author. The first Kaggle run is the test:
start with the three test shots (see README).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

KIND_TO_ENGINE = {"ref": "refs", "keyframe": "keyframe", "variant": "keyframe", "upscale": "upscale"}


# ---------------------------------------------------------------- shared helpers
def wan_frames(seconds: float, fps: int, max_frames: int) -> int:
    """Wan needs 4k+1 frames. Smallest such count covering the shot, capped by the model."""
    need = max(1, math.ceil(seconds * fps))
    frames = ((need - 1 + 3) // 4) * 4 + 1
    cap = ((max_frames - 1) // 4) * 4 + 1
    return max(5, min(frames, cap))


def wan_segments(seconds: float, fps: int, max_frames: int) -> list[int]:
    """Frame counts of the clips that cover a shot. Each is 4k+1 frames and at most max_frames.

    Long shots are chained: each next clip starts from the previous clip's last frame,
    so it shares one frame with it. Total new frames >= the shot's length.
    """
    need = max(1, math.ceil(seconds * fps))
    first = wan_frames(seconds, fps, max_frames)
    segs, have = [first], first
    while have < need:
        nxt = wan_frames((need - have + 1) / fps, fps, max_frames)
        segs.append(nxt)
        have += nxt - 1
    return segs


def cover_crop_box(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int, int, int]:
    """Scale to cover dst exactly, then centre-crop. Returns (scaled_w, scaled_h, left, top)."""
    scale = max(dst_w / src_w, dst_h / src_h)
    sw, sh = max(dst_w, round(src_w * scale)), max(dst_h, round(src_h * scale))
    return sw, sh, (sw - dst_w) // 2, (sh - dst_h) // 2


def seed_for(key: str) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg  # noqa: PLC0415
    return imageio_ffmpeg.get_ffmpeg_exe()


LOG_FILE = None          # set by run(): <out>/worker.jsonl travels back with the results


def log(msg: str, event: str = "note", level: str = "info", **fields) -> None:
    """Structured log line, same layout as engine/log.py: time, level, component, event, message,
    key=value fields; also one JSON object per line in LOG_FILE."""
    now = time.time()
    kv = "  ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    print(f"{time.strftime('%H:%M:%S', time.localtime(now))} {level.upper():5} worker {event:16} {msg}"
          + (f"  {kv}" if kv else ""), flush=True)
    if LOG_FILE is not None:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "level": level,
                                    "comp": "worker", "event": event, "msg": msg, **fields}, default=str) + "\n")
        except OSError:
            pass


# ---------------------------------------------------------------- stub engines (CPU, for tests)
class StubImage:
    """Solid images at the configured size. Tests the contract, not the art."""

    def __init__(self, cfg: dict):
        self.w, self.h = cfg.get("width", 1664), cfg.get("height", 928)

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        for ref in job.get("refs", []) + ([job["image"]] if job.get("image") else []):
            if not (inputs / ref).exists():
                raise FileNotFoundError(f"input {ref} missing from the batch")
        w, h = job.get("width", self.w), job.get("height", self.h)
        subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        f"color=c=0x4a5568:s={w}x{h}", "-frames:v", "1", str(out)], check=True)
        return {"width": w, "height": h}


class StubVideo:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        c = self.cfg
        segs = wan_segments(job["seconds"], c["fps"], c["max_frames"])
        frames = sum(segs) - (len(segs) - 1)
        img = inputs / job["image"]
        subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", "-loop", "1", "-i", str(img),
                        "-frames:v", str(frames), "-r", str(c["fps"]),
                        "-vf", f"scale={c['width']}:{c['height']},format=yuv420p", str(out)], check=True)
        return {"frames": frames, "segments": segs, "fps": c["fps"], "width": c["width"], "height": c["height"]}


class StubUpscale:
    def __init__(self, cfg: dict):
        self.w, self.h = cfg["target_width"], cfg["target_height"]

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        src = inputs / job["source"]
        vf = f"scale={self.w}:{self.h}:force_original_aspect_ratio=increase,crop={self.w}:{self.h}"
        if job["media"] == "video":
            args = ["-i", str(src), "-vf", vf + ",format=yuv420p", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "30"]
        else:
            args = ["-i", str(src), "-vf", vf, "-frames:v", "1"]
        subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", *args, str(out)], check=True)
        return {"width": self.w, "height": self.h}


# ---------------------------------------------------------------- GPU engines
def pick_dtype(name: str):
    import torch  # noqa: PLC0415
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16          # T4 (Turing) has no native bf16


def find_gguf(repo: str, subdir: str, prefer: list[str], max_gb: float | None = None) -> str | None:
    """Download the first GGUF in `prefer` order that the repo has and that fits in `max_gb`
    (the GPU memory it may take). If none of `prefer` fits, the largest GGUF that does.
    None if the repo has none that fits."""
    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415
    try:
        files = {f.path: (f.size or 0) / 2**30 for f in HfApi().list_repo_tree(repo, recursive=True)
                 if getattr(f, "size", None) is not None and f.path.endswith(".gguf")
                 and (not subdir or f.path.startswith(subdir + "/"))}
    except Exception as e:  # noqa: BLE001
        log(f"GGUF repo {repo} not reachable ({e}); falling back")
        return None
    fits = {p: gb for p, gb in files.items() if max_gb is None or gb <= max_gb}
    choice = next((p for q in prefer for p in sorted(fits) if q in p), None)
    if choice is None and fits:
        choice = max(fits, key=fits.get)
    if choice is None:
        log(f"no GGUF in {repo}/{subdir} fits {max_gb:.1f} GB; sizes: "
            + ", ".join(f"{Path(p).name} {gb:.1f}" for p, gb in sorted(files.items(), key=lambda x: x[1])))
        return None
    log(f"using {repo}/{choice} ({files[choice]:.1f} GB" + (f", room {max_gb:.1f} GB)" if max_gb else ")"))
    return hf_hub_download(repo, choice)


def gpu_room_gb(headroom: float) -> float | None:
    """GPU memory a model's weights may take on GPU 0, leaving `headroom` GB for activations."""
    import torch  # noqa: PLC0415
    if not torch.cuda.is_available():
        return None
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    log(f"GPU 0: {torch.cuda.get_device_name(0)}, {total:.1f} GB; {torch.cuda.device_count()} GPU(s)")
    return total - headroom


def snapshot(model: str, skip: list[str]) -> str:
    """Download a diffusers repo, leaving out the weights of the `skip` folders (they come from a
    GGUF instead). Their config.json is still needed to build the model around the GGUF."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    ignore = [f"{d}/*{ext}" for d in skip for ext in (".safetensors", ".bin", ".pt", ".pth", ".gguf")]
    return snapshot_download(model, ignore_patterns=ignore)


def check_not_blank(arr, what: str) -> None:
    import numpy as np  # noqa: PLC0415
    a = np.asarray(arr, dtype="float32")
    if not np.isfinite(a).all() or float(a.max()) - float(a.min()) < 0.02:
        raise RuntimeError(f"{what} came out blank or NaN; this GPU may not handle the chosen dtype. "
                           "Try engines.dtype = 'fp32' for this stage, or a smaller quantisation.")


class QwenKeyframes:
    """Qwen-Image-Edit-2511: reference stills in, a new frame out. 1 to 3 references."""

    PIPELINE = "QwenImageEditPlusPipeline"

    def __init__(self, cfg: dict, dtype_name: str):
        import diffusers  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from diffusers import (GGUFQuantizationConfig, PipelineQuantizationConfig,  # noqa: PLC0415
                               QwenImageTransformer2DModel)
        self.cfg, self.torch = cfg, torch
        dtype = pick_dtype(dtype_name)
        g = cfg.get("transformer_gguf") or {}
        gguf = find_gguf(g["repo"], g.get("dir", ""), g.get("prefer", [])) if g else None
        kwargs = {"torch_dtype": dtype}
        quantize = ["text_encoder"]
        if gguf:
            local = snapshot(cfg["model"], skip=["transformer"])
            kwargs["transformer"] = QwenImageTransformer2DModel.from_single_file(
                gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                config=local, subfolder="transformer", torch_dtype=dtype)
        else:
            local = snapshot(cfg["model"], skip=[])
            quantize.append("transformer")
        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": dtype},
            components_to_quantize=quantize)
        self.pipe = getattr(diffusers, self.PIPELINE).from_pretrained(local, **kwargs)
        self.pipe.enable_model_cpu_offload()
        self.pipe.set_progress_bar_config(disable=True)

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        from PIL import Image  # noqa: PLC0415
        c = self.cfg
        if job.get("image"):                                   # a mouth-shape variant edits the keyframe
            images = [Image.open(inputs / job["image"]).convert("RGB")]
        else:
            images = [Image.open(inputs / r).convert("RGB") for r in job.get("refs", [])][: c.get("max_refs", 3)]
        if not images:
            raise RuntimeError("no reference stills for this shot: add them to style_bible/refs (gate 1)")
        with self.torch.inference_mode():
            result = self.pipe(image=images, prompt=job["prompt"], negative_prompt=c.get("negative_prompt", " "),
                               true_cfg_scale=c.get("true_cfg_scale", 4.0), guidance_scale=1.0,
                               num_inference_steps=c.get("steps", 40), width=c["width"], height=c["height"],
                               generator=self.torch.Generator("cpu").manual_seed(seed_for(job["key"])))
        img = result.images[0]
        check_not_blank(img, "keyframe")
        img.save(out)
        return {"width": img.width, "height": img.height, "seed": seed_for(job["key"])}


LIGHTNING_SCHEDULER = {   # from ModelTC/Qwen-Image-Lightning's diffusers example
    "base_image_seq_len": 256, "base_shift": 1.0986122886681098, "invert_sigmas": False,
    "max_image_seq_len": 8192, "max_shift": 1.0986122886681098, "num_train_timesteps": 1000, "shift": 1.0,
    "shift_terminal": None, "stochastic_sampling": False, "time_shift_type": "exponential",
    "use_beta_sigmas": False, "use_dynamic_shifting": True, "use_exponential_sigmas": False,
    "use_karras_sigmas": False}


class QwenStills:
    """Qwen-Image-2512 text-to-image: the candidate reference stills for gate 1.

    Sized for a 15 GB T4: the prompts are encoded first on the CPU (bf16) and the 7B text encoder
    is freed; then only the transformer (the largest GGUF that fits this GPU) and the VAE sit on
    the GPU, with no offloading. An optional few-step LoRA (engines.refs.lora) cuts 50 steps with
    CFG to 4-8 steps without; if it will not load, the base settings are used."""

    def __init__(self, cfg: dict, dtype_name: str, prompts: list[str], negative: str):
        import diffusers  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from diffusers import GGUFQuantizationConfig, QwenImageTransformer2DModel  # noqa: PLC0415
        self.cfg, self.torch = cfg, torch
        dtype = pick_dtype(dtype_name)
        g = cfg.get("transformer_gguf") or {}
        local = snapshot(cfg["model"], skip=["transformer"] if g else [])

        # 1. every prompt of the batch, on the CPU
        tp = diffusers.QwenImagePipeline.from_pretrained(local, transformer=None, vae=None, torch_dtype=torch.bfloat16)
        enc = torch.device("cpu")
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and \
                torch.cuda.get_device_properties(0).total_memory / 2**30 >= 30:
            enc = torch.device("cuda")                  # A100 and up: seconds instead of minutes
            tp.to(enc)
        self.embeds = {}
        t0 = time.time()
        for text in dict.fromkeys(prompts + [negative]):
            t1 = time.time()
            with torch.inference_mode():
                e, m = tp.encode_prompt(prompt=text, device=enc)
            self.embeds[text] = (e.cpu(), None if m is None else m.cpu())   # m is None when every token counts
            log(f"prompt {len(self.embeds)}/{len(set(prompts + [negative]))} encoded on the {enc.type} in {time.time() - t1:.0f} s")
        log(f"encoded {len(self.embeds)} prompts on the {enc.type} in {time.time() - t0:.0f} s")
        del tp
        gc.collect()
        if enc.type == "cuda":
            torch.cuda.empty_cache()

        # 2. the transformer, sized to this GPU, and the VAE: nothing else on the GPU
        room = gpu_room_gb(float(cfg.get("vram_headroom_gb", 2.5)))
        gguf = find_gguf(g["repo"], g.get("dir", ""), g.get("prefer", []), room) if g else None
        kwargs = {"torch_dtype": dtype, "text_encoder": None, "tokenizer": None}
        if gguf:
            kwargs["transformer"] = QwenImageTransformer2DModel.from_single_file(
                gguf, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                config=local, subfolder="transformer", torch_dtype=dtype)
        elif g:
            raise RuntimeError(f"no GGUF of {g['repo']} fits this GPU ({room} GB)")
        self.pipe = diffusers.QwenImagePipeline.from_pretrained(local, **kwargs)
        self.steps, self.cfg_scale = cfg.get("steps", 50), cfg.get("true_cfg_scale", 4.0)
        self._lora(cfg.get("lora"))
        if gguf:
            self.pipe.to("cuda")
        else:
            self.pipe.enable_model_cpu_offload()
        if cfg.get("vae_fp32", True):                  # the decoder is the classic fp16 overflow; 0.25 GB in fp32
            self.pipe.vae.to(dtype=torch.float32)
        try:
            self.pipe.vae.enable_tiling()
        except Exception:  # noqa: BLE001
            pass
        log(f"stills engine: transformer {dtype}, VAE {self.pipe.vae.dtype}, {self.steps} steps, "
            f"cfg {self.cfg_scale}; GPU memory in use {torch.cuda.memory_allocated() / 2**30:.1f} GB")
        self.first = True
        self.pipe.set_progress_bar_config(disable=True)
        self.negative = negative

    def _lora(self, lora: dict | None) -> None:
        if not lora or not lora.get("repo"):
            return
        from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: PLC0415
        from huggingface_hub import list_repo_files  # noqa: PLC0415
        name = None
        try:
            files = [f for f in list_repo_files(lora["repo"]) if f.endswith(".safetensors")]
            log(f"LoRA files in {lora['repo']}: {files}")
            name = next((f for q in lora.get("prefer", []) for f in files if q in f), files[0] if files else None)
            self.pipe.load_lora_weights(lora["repo"], weight_name=name)
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
            self.steps, self.cfg_scale = lora.get("steps", 8), lora.get("true_cfg_scale", 1.0)
            log(f"few-step LoRA {lora['repo']}/{name}: {self.steps} steps, cfg {self.cfg_scale}")
        except Exception as e:  # noqa: BLE001 - slower but still works without it
            try:
                self.pipe.unload_lora_weights()
            except Exception:  # noqa: BLE001
                pass
            log(f"LoRA {lora['repo']}/{name} not used; {self.steps} steps, cfg "
                f"{self.cfg_scale}. Error: {e!r}"[:900])
            log(traceback.format_exc()[-1200:])

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        torch = self.torch
        w, h = job.get("width", 1664), job.get("height", 928)
        dev, dt = "cuda", self.pipe.transformer.dtype
        e, m = self.embeds[job["prompt"]]
        args = dict(prompt_embeds=e.to(dev, dt), prompt_embeds_mask=None if m is None else m.to(dev),
                    num_inference_steps=self.steps, true_cfg_scale=self.cfg_scale, width=w, height=h,
                    generator=torch.Generator("cpu").manual_seed(seed_for(job["key"])))
        if self.cfg_scale > 1:
            ne, nm = self.embeds[self.negative]
            args.update(negative_prompt_embeds=ne.to(dev, dt),
                        negative_prompt_embeds_mask=None if nm is None else nm.to(dev))
        watch, t_step = self.first, [time.time()]
        self.first = False

        def nan_guard(pipe, step, t, kw):              # stop at the first NaN step, not after all of them
            lat = kw.get("latents")
            if lat is not None and not torch.isfinite(lat).all():
                raise RuntimeError(f"reference still came out blank or NaN at step {step + 1} (in the "
                                   f"transformer, before the decoder); latents {lat.dtype}")
            if watch and lat is not None:
                log(f"  step {step + 1}/{self.steps}: {time.time() - t_step[0]:.1f} s, "
                    f"latents max |x| {float(lat.abs().max()):.2f}")
                t_step[0] = time.time()
            return kw

        args.update(callback_on_step_end=nan_guard, callback_on_step_end_tensor_inputs=["latents"])
        t0 = time.time()
        try:
            with torch.inference_mode():
                img = self.pipe(**args).images[0]
        finally:
            torch.cuda.empty_cache()
        check_not_blank(img, "reference still (latents were fine, so in the decoder)")
        img.save(out)
        return {"width": img.width, "height": img.height, "seed": seed_for(job["key"]),
                "steps": self.steps, "seconds": round(time.time() - t0, 1)}


class WanVideo:
    """Wan 2.2 image-to-video. Prompts are encoded first on the CPU, then the encoder is freed,
    so the text encoder and both 14B experts never sit in RAM together."""

    def __init__(self, cfg: dict, dtype_name: str, prompts: list[str], negative: str):
        import torch  # noqa: PLC0415
        from diffusers import (AutoencoderKLWan, GGUFQuantizationConfig,  # noqa: PLC0415
                               WanImageToVideoPipeline, WanTransformer3DModel)
        from transformers import AutoTokenizer, UMT5EncoderModel  # noqa: PLC0415
        self.cfg, self.torch = cfg, torch
        dtype = pick_dtype(dtype_name)
        g = cfg.get("transformer_gguf")
        skip = ["transformer", "transformer_2"] if g else []
        local = snapshot(cfg["model"], skip=skip)

        # 1. text encoding on the CPU in bf16 (T5-class encoders overflow in fp16)
        tok = AutoTokenizer.from_pretrained(local, subfolder="tokenizer")
        enc = UMT5EncoderModel.from_pretrained(local, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        self.embeds = {}
        for text in list(dict.fromkeys(prompts + [negative])):
            self.embeds[text] = self._encode(tok, enc, text)
        del enc
        gc.collect()

        # 2. the video pipeline without a text encoder
        kwargs = {"torch_dtype": dtype, "text_encoder": None, "tokenizer": None,
                  "vae": AutoencoderKLWan.from_pretrained(local, subfolder="vae", torch_dtype=torch.float32)}
        if g:
            q = GGUFQuantizationConfig(compute_dtype=dtype)
            hi = find_gguf(g["repo"], g["high_noise_dir"], g["prefer"])
            lo = find_gguf(g["repo"], g["low_noise_dir"], g["prefer"])
            if not hi or not lo:
                raise RuntimeError(f"GGUF weights not found in {g['repo']}")
            kwargs["transformer"] = WanTransformer3DModel.from_single_file(
                hi, quantization_config=q, config=local, subfolder="transformer", torch_dtype=dtype)
            kwargs["transformer_2"] = WanTransformer3DModel.from_single_file(
                lo, quantization_config=q, config=local, subfolder="transformer_2", torch_dtype=dtype)
        if cfg.get("expand_timesteps"):
            kwargs["expand_timesteps"] = True
        self.pipe = WanImageToVideoPipeline.from_pretrained(local, **kwargs)
        self.pipe.enable_model_cpu_offload()
        try:
            self.pipe.vae.enable_tiling()
        except Exception:  # noqa: BLE001 - tiling is an optimisation only
            pass
        self.pipe.set_progress_bar_config(disable=True)
        self.negative = negative

    def _encode(self, tok, enc, text: str):
        """Same as diffusers' Wan _get_t5_prompt_embeds, on the CPU."""
        torch = self.torch
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean  # noqa: PLC0415
        t = tok([prompt_clean(text)], padding="max_length", max_length=512, truncation=True,
                add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        with torch.inference_mode():
            h = enc(t.input_ids, t.attention_mask).last_hidden_state
        n = int(t.attention_mask.gt(0).sum())
        e = torch.zeros_like(h)
        e[:, :n] = h[:, :n]
        return e

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        torch, c = self.torch, self.cfg
        segs = wan_segments(job["seconds"], c["fps"], c["max_frames"])
        img = Image.open(inputs / job["image"]).convert("RGB")
        sw, sh, left, top = cover_crop_box(img.width, img.height, c["width"], c["height"])
        img = img.resize((sw, sh), Image.LANCZOS).crop((left, top, left + c["width"], top + c["height"]))
        clips = []
        for n, frames in enumerate(segs):
            args = dict(image=img, prompt_embeds=self.embeds[job["prompt"]],
                        negative_prompt_embeds=self.embeds[self.negative],
                        height=c["height"], width=c["width"], num_frames=frames,
                        num_inference_steps=c.get("steps", 40), guidance_scale=c.get("guidance_scale", 5.0),
                        generator=torch.Generator("cpu").manual_seed(seed_for(job["key"]) + n),
                        output_type="np")
            if "guidance_scale_2" in c:
                args["guidance_scale_2"] = c["guidance_scale_2"]
            with torch.inference_mode():
                video = self.pipe(**args).frames[0]        # (frames, h, w, 3) floats 0..1
            check_not_blank(video, "video")
            video = (np.clip(video, 0, 1) * 255).round().astype("uint8")
            clips.append(video if n == 0 else video[1:])  # the next clip starts on the last frame
            img = Image.fromarray(video[-1])
        full = np.concatenate(clips, axis=0)
        write_video(full, out, c["fps"], crf=10)
        return {"frames": int(full.shape[0]), "segments": segs, "fps": c["fps"], "width": c["width"],
                "height": c["height"], "seed": seed_for(job["key"])}


def write_video(frames, out: Path, fps: int, crf: int) -> None:
    """uint8 frames (n, h, w, 3) to H.264 at high quality."""
    h, w = frames.shape[1:3]
    p = subprocess.Popen([ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                          "-crf", str(crf), "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed to write the video")


class Upscale4K:
    """Upscale, then an exact cover-resize and centre crop to 3840x2160.

    Takes use a video-stable model so frames do not flicker; stills use a sharper
    still-image model. Each is loaded the first time it is needed.
    """

    def __init__(self, cfg: dict, dtype_name: str):
        import torch  # noqa: PLC0415
        self.cfg, self.torch, self.dtype_name = cfg, torch, dtype_name
        self.models: dict[str, tuple] = {}
        self.model, self.half = None, False

    def _load(self, media: str):
        if media in self.models:
            self.model, self.half = self.models[media]
            return
        torch = self.torch
        from spandrel import ImageModelDescriptor, ModelLoader  # noqa: PLC0415
        w = self.cfg[f"{media}_weights"]
        model = ModelLoader().load_from_file(self._weights(w))
        if not isinstance(model, ImageModelDescriptor):
            raise RuntimeError("upscale weights are not an image-to-image model")
        half = bool(model.supports_half) and pick_dtype(self.dtype_name) == torch.float16
        model.cuda().eval()
        if half:
            model.half()
        log(f"{media} upscaler {model.architecture} x{model.scale}, half={half}")
        self.models[media] = (model, half)
        self.model, self.half = model, half

    @staticmethod
    def _weights(w: dict) -> str:
        import urllib.request  # noqa: PLC0415

        def fetch(url: str) -> str:
            dest = Path("/tmp") / Path(url).name
            if not dest.exists():
                urllib.request.urlretrieve(url, dest)
            return str(dest)

        if w.get("repo"):
            try:
                from huggingface_hub import hf_hub_download  # noqa: PLC0415
                return hf_hub_download(w["repo"], w["file"], repo_type=w.get("repo_type", "model"))
            except Exception as e:  # noqa: BLE001
                if not w.get("fallback_url"):
                    raise
                log(f"weights {w['repo']} unavailable ({e}); using {w['fallback_url']}")
                return fetch(w["fallback_url"])
        return fetch(w["url"])

    def _frame(self, rgb):
        """uint8 HxWx3 -> uint8 3840x2160x3."""
        torch = self.torch
        import torch.nn.functional as F  # noqa: PLC0415
        tw, th = self.cfg["target_width"], self.cfg["target_height"]
        x = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float() / 255.0
        if self.half:
            x = x.half()
        with torch.inference_mode():
            y = self.model(x).float().clamp(0, 1)
            sw, sh, left, top = cover_crop_box(y.shape[3], y.shape[2], tw, th)
            if (sw, sh) != (y.shape[3], y.shape[2]):
                y = F.interpolate(y, size=(sh, sw), mode="bicubic", align_corners=False, antialias=True)
            y = y[:, :, top:top + th, left:left + tw].clamp(0, 1)
        return (y[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        src = inputs / job["source"]
        tw, th = self.cfg["target_width"], self.cfg["target_height"]
        self._load(job["media"])
        if job["media"] == "image":
            Image.fromarray(self._frame(np.asarray(Image.open(src).convert("RGB")))).save(out)
            return {"width": tw, "height": th}
        import imageio_ffmpeg  # noqa: PLC0415
        reader = imageio_ffmpeg.read_frames(str(src), pix_fmt="rgb24")
        meta = next(reader)
        w, h = meta["size"]
        fps = meta.get("fps") or 24
        writer = subprocess.Popen([ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "-s", f"{tw}x{th}", "-r", str(fps), "-i", "-",
                                   "-c:v", "libx264", "-preset", "veryfast", "-crf", str(self.cfg.get("crf", 12)),
                                   "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
        n = 0
        for raw in reader:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            writer.stdin.write(self._frame(frame).tobytes())
            n += 1
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError("ffmpeg failed to write the 4K video")
        return {"width": tw, "height": th, "frames": n, "fps": fps}


# ---------------------------------------------------------------- generators
# What makes each group of jobs. A group is named after its engine in the manifest ("refs",
# "keyframe", "upscale") or, for takes, after the shot's video model ("Wan 2.2 14B"). Each generator
# is (the real one, its stand-in with no model).
def _refs(cfg: dict, dtype: str, jobs: list[dict], engines: dict):
    return QwenStills(cfg, dtype, [j["prompt"] for j in jobs], cfg.get("negative_prompt", " "))


def _keyframe(cfg: dict, dtype: str, jobs: list[dict], engines: dict):
    return QwenKeyframes(cfg, dtype)


def _video(cfg: dict, dtype: str, jobs: list[dict], engines: dict):
    return WanVideo(cfg, dtype, [j["prompt"] for j in jobs], engines.get("negative_prompt", ""))


def _upscale(cfg: dict, dtype: str, jobs: list[dict], engines: dict):
    return Upscale4K(cfg, dtype)


GENERATORS = {"refs": (_refs, StubImage), "keyframe": (_keyframe, StubImage), "video": (_video, StubVideo),
              "upscale": (_upscale, StubUpscale)}


def generator_for(name: str, jobs: list[dict]) -> str:
    """Which generator makes a group: by its name, or "video" for a group of takes. Anything else
    is an error, never a guess: a wrong generator would spend GPU time on the wrong model."""
    if name == "refs" or name.startswith("refs_"):
        return "refs"
    if name in ("keyframe", "upscale"):
        return name
    if jobs and all(j["kind"] == "take" for j in jobs):
        return "video"
    raise ValueError(f"no generator makes the {name!r} group ({', '.join(sorted({j['kind'] for j in jobs}))} jobs)")


# ---------------------------------------------------------------- the run
def build_engine(name: str, jobs: list[dict], engines: dict, engine_set: str):
    dtype = (engines.get(name) or {}).get("dtype") or engines.get("dtype", "auto")
    real, stand_in = GENERATORS[generator_for(name, jobs)]
    cfg = engines[name]
    return stand_in(cfg) if engine_set == "stub" else real(cfg, dtype, jobs, engines)


def free_engine(engine) -> None:
    del engine
    gc.collect()
    try:
        import torch  # noqa: PLC0415
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def run(manifest: Path, inputs: Path, out: Path, engine_set: str, expect_batch: str | None = None) -> dict:
    global LOG_FILE
    t0 = time.time()
    m = json.loads(manifest.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    LOG_FILE = out / "worker.jsonl"
    log(f"batch {m['batch_id']}: {len(m['jobs'])} job(s)", event="batch.start", batch=m["batch_id"],
        jobs=len(m["jobs"]), stage=m.get("stage"), engines=engine_set)
    results = {"batch_id": m["batch_id"], "engine_set": engine_set, "started": t0, "jobs": {},
               "groups": {}, "error": None}

    def save():
        results["elapsed"] = time.time() - t0
        (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    if expect_batch and m["batch_id"] != expect_batch:
        results["error"] = f"stale inputs: kernel expected {expect_batch}, dataset has {m['batch_id']}"
        save()
        return results
    deadline = t0 + float(m.get("deadline_seconds", 10 * 3600))
    groups: dict[str, list[dict]] = {}
    for job in m["jobs"]:
        name = job.get("engine") or ("keyframe" if job["kind"] == "ref" and job.get("refs")
                                     else KIND_TO_ENGINE.get(job["kind"]) or job.get("model"))
        groups.setdefault(name, []).append(job)
    for name, jobs in groups.items():
        if time.time() + 600 > deadline:
            break
        g0 = time.time()
        try:
            log(f"loading engine {name} for {len(jobs)} jobs", event="engine.load", engine=name, jobs=len(jobs))
            engine = build_engine(name, jobs, m["engines"], engine_set)
        except Exception as e:  # noqa: BLE001
            log(f"engine {name} failed to load: {e!r}"[:600], event="engine.failed", level="error", engine=name)
            log(traceback.format_exc()[-1500:])
            for job in jobs:
                results["jobs"][job["key"]] = {"ok": False, "error": f"engine {name} failed to load: {e!r}"[:800]}
            results["groups"][name] = {"load_error": traceback.format_exc()[-2000:]}
            save()
            continue
        results["groups"][name] = {"load_seconds": time.time() - g0}
        log(f"engine {name} ready in {time.time() - g0:.0f} s")
        slow = 0
        broken = 0                                         # machine failures in a row (blank, NaN, CUDA)
        last_err = ""
        factor = float(m.get("slow_factor", 3))
        for n, job in enumerate(jobs):
            est = float(job.get("est_seconds", 0))
            if time.time() + est > deadline:
                continue                                   # left for the next session
            if broken >= 2:
                log(f"stopping {name}: 2 jobs in a row failed the same machine way ({last_err[:120]}); "
                    f"{len(jobs) - n} left for after a fix, so no more GPU time goes on it",
                    event="group.stopped_broken", level="error", engine=name, left=len(jobs) - n)
                results["groups"][name]["stopped_broken"] = len(jobs) - n
                break
            if slow >= 2:
                log(f"stopping {name}: 2 jobs took over {factor:g}x their estimate; "
                    f"{len(jobs) - n} left for the next batch", event="group.stopped_slow", level="warn",
                    engine=name, left=len(jobs) - n)
                results["groups"][name]["stopped_slow"] = len(jobs) - n
                break
            j0 = time.time()
            ext = ".mp4" if job["kind"] == "take" or job.get("media") == "video" else ".png"
            target = out / f"{job['key']}{ext}"
            try:
                info = engine.run(job, inputs, target)
                took = time.time() - j0
                results["jobs"][job["key"]] = {"ok": True, "file": target.name,
                                               "gpu_seconds": took, **(info or {})}
                log(f"{job['key']} ok in {took:.0f} s (estimate {est:.0f} s) [{n + 1}/{len(jobs)}]", event="job.ok",
                    key=job["key"], seconds=round(took), estimate=round(est), step=f"{n + 1}/{len(jobs)}")
                broken = 0
                if est > 0 and took > factor * est:
                    slow += 1
            except Exception as e:  # noqa: BLE001
                oom = "out of memory" in str(e).lower()
                results["jobs"][job["key"]] = {"ok": False, "error": repr(e)[:800], "oom": oom,
                                               "gpu_seconds": time.time() - j0}
                log(f"{job['key']} failed: {e!r}", event="job.failed", level="error", key=job["key"],
                    oom=oom, seconds=round(time.time() - j0))
                last_err = str(e)
                if oom or any(k in last_err for k in ("blank or NaN", "CUDA", "same device")):
                    broken += 1
                else:
                    broken = 0
                if oom:
                    free_engine(None)
            save()
        free_engine(engine)
        engine = None
    save()
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--engine-set", default="gpu", choices=["gpu", "stub"])
    ap.add_argument("--expect-batch")
    a = ap.parse_args(argv)
    r = run(a.manifest, a.inputs, a.out, a.engine_set, a.expect_batch)
    ok = sum(1 for v in r["jobs"].values() if v.get("ok"))
    log(f"done: {ok}/{len(r['jobs'])} jobs ok in {r['elapsed'] / 60:.1f} min", event="batch.done",
        ok=ok, jobs=len(r["jobs"]), minutes=round(r["elapsed"] / 60, 1))
    return 0 if r.get("error") is None else 3


if __name__ == "__main__":
    sys.exit(main())
